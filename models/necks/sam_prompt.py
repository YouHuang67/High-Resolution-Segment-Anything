import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.utils.misc import to_2tuple
from mmseg.models.builder import NECKS

from engine.utils import rearrange, repeat


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).square().mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = rearrange(self.weight, 'c -> c () ()') * x + \
            rearrange(self.bias, 'c -> c () ()')
        return x


class PositionEmbedRandom(nn.Module):

    def __init__(self, embed_dim=128, scale=None):
        super(PositionEmbedRandom, self).__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, embed_dim // 2)))

    def forward_with_size(self, size):
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        ys = (grid.cumsum(dim=0) - 0.5) / h
        xs = (grid.cumsum(dim=1) - 0.5) / w
        pos_embed = self.encode_position(torch.stack([xs, ys], dim=-1))
        pos_embed = rearrange(pos_embed, "h w c -> c h w")
        return pos_embed

    def forward_with_coords(self, coords_input, image_size):
        coords = coords_input.clone()
        coords[..., 0] = coords[..., 0] / image_size[1]
        coords[..., 1] = coords[..., 1] / image_size[0]
        return self.encode_position(coords.to(torch.float))  # B x N x C

    def encode_position(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)


@NECKS.register_module()
class SAMPromptEncoder(nn.Module):

    def __init__(self,
                 embed_dim,
                 image_embed_size,
                 input_image_size,
                 mask_in_dim,
                 activation=nn.GELU,
                 pos_embed_cls=PositionEmbedRandom):
        super(SAMPromptEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.input_image_size = to_2tuple(input_image_size)
        self.image_embed_size = to_2tuple(image_embed_size)
        self.pos_embed_layer = pos_embed_cls(embed_dim=embed_dim)

        self.num_point_embeds = 4  # pos/neg point + 2 box corners
        self.point_embeds = nn.ModuleList([
            nn.Embedding(1, embed_dim)
            for _ in range(self.num_point_embeds)])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embed_size[0],
                                4 * image_embed_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_dim // 4),
            activation(),
            nn.Conv2d(mask_in_dim // 4, mask_in_dim, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_dim),
            activation(),
            nn.Conv2d(mask_in_dim, embed_dim, kernel_size=1))
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def encode_prompts(self, image_size, embed_size,
                       points, boxes=None, prev_logits=None):
        if points is not None:
            batch_size = points[0].shape[0]
        elif boxes is not None:
            batch_size = boxes.shape[0]
        elif prev_logits is not None:
            batch_size = prev_logits.shape[0]
        else:
            raise ValueError("Not found any prompts")
        sparse_embeds = torch.empty(
            (batch_size, 0, self.embed_dim), device=self.device)
        if points is not None:
            coords, labels = points
            point_embeds = self.embed_points(
                image_size, coords, labels, pad=(boxes is None))
            sparse_embeds = torch.cat([sparse_embeds, point_embeds], dim=1)
        if boxes is not None:
            box_embeds = self.embed_boxes(image_size, boxes)
            sparse_embeds = torch.cat([sparse_embeds, box_embeds], dim=1)

        if prev_logits is not None:
            dense_embeds = self.embed_logits(prev_logits)
        else:
            dense_embeds = self.no_mask_embed.weight
            dense_embeds = repeat(dense_embeds, "() d -> b d h w",
                                  b=batch_size,
                                  h=self.image_embed_size[0],
                                  w=self.image_embed_size[1])
        if tuple(dense_embeds.shape[-2:]) != embed_size:
            dense_embeds = F.interpolate(
                dense_embeds, embed_size, mode='bilinear')
        image_pos_embeds = self.image_pos_embeds
        if tuple(image_pos_embeds.shape[-2:]) != embed_size:
            image_pos_embeds = F.interpolate(
                image_pos_embeds, embed_size, mode='bilinear')
        return dict(sparse_embeds=sparse_embeds,
                    dense_embeds=dense_embeds,
                    image_pos_embeds=image_pos_embeds)

    def forward(self, image_embeds, points, boxes=None, prev_logits=None):
        ori_image_embeds = image_embeds
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = image_embeds[-1]
        image_size = (
            image_embeds.size(-2) *
            self.input_image_size[0] // self.image_embed_size[0],
            image_embeds.size(-1) *
            self.input_image_size[1] // self.image_embed_size[1])
        embed_size = (
            image_size[0] //
            (self.input_image_size[0] // self.image_embed_size[0]),
            image_size[1] //
            (self.input_image_size[1] // self.image_embed_size[1]))
        return dict(
            ori_image_embeds=ori_image_embeds,
            image_embeds=image_embeds,
            **self.encode_prompts(image_size, embed_size,
                                  points, boxes, prev_logits)
        )

    @property
    def device(self):
        return self.point_embeds[0].weight.device

    def embed_points(self, input_image_size, points, labels, pad):
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros(
                (points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones(
                (labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embed = \
            self.pos_embed_layer.forward_with_coords(points,
                                                     input_image_size)
        point_embed[labels == -1] = 0.0
        point_embed[labels == -1] += self.not_a_point_embed.weight
        point_embed[labels == 0] += self.point_embeds[0].weight
        point_embed[labels == 1] += self.point_embeds[1].weight
        return point_embed

    def embed_boxes(self, input_image_size, boxes):
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embed = \
            self.pos_embed_layer.forward_with_coords(coords,
                                                     input_image_size)
        corner_embed[:, 0, :] += self.point_embeds[2].weight
        corner_embed[:, 1, :] += self.point_embeds[3].weight
        return corner_embed

    def embed_logits(self, masks):
        return self.mask_downscaling(masks)

    @property
    def image_pos_embeds(self):
        embeds = self.pos_embed_layer.forward_with_size(self.image_embed_size)
        embeds = rearrange(embeds, "c h w -> () c h w")
        return embeds

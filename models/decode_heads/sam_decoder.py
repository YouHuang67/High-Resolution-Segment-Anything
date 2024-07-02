import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import HEADS

from engine.utils import rearrange, repeat
from engine.utils.xformers import memory_efficient_attention
from engine.decode_heads import BaseDecodeHead


class TwoLayerMLP(nn.Module):

    def __init__(self, embed_dim, mlp_dim, act_layer=nn.GELU):
        super(TwoLayerMLP, self).__init__()
        self.lin1 = nn.Linear(embed_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embed_dim)
        self.act = act_layer()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in
            zip([input_dim] + dims, dims + [output_dim])])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x) if i < self.num_layers - 1 else x
        return x


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).square().mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = rearrange(self.weight, 'c -> c () ()') * x + \
            rearrange(self.bias, 'c -> c () ()')
        return x


class QKVSeparateAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, downsample_rate=1):
        super(QKVSeparateAttention, self).__init__()
        if (embed_dim // downsample_rate) % num_heads != 0:
            raise ValueError(
                f"each head`s embed_dim {embed_dim // downsample_rate} "
                f"cannot be divided by downsample_rate {downsample_rate}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.downsample_rate = downsample_rate
        self.q_proj = nn.Linear(embed_dim, embed_dim // downsample_rate)
        self.k_proj = nn.Linear(embed_dim, embed_dim // downsample_rate)
        self.v_proj = nn.Linear(embed_dim, embed_dim // downsample_rate)
        self.scale = (embed_dim // downsample_rate // num_heads) ** -0.5
        self.out_proj = nn.Linear(embed_dim // downsample_rate, embed_dim)

    def forward(self, q, k, v):
        q = rearrange(self.q_proj(q), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(k), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(v), 'b n (h d) -> b h n d', h=self.num_heads)
        out = memory_efficient_attention(q, k, v, scale=self.scale)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        return out


class CrossAttentionBlock(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_dim=2048,
                 act_layer=nn.ReLU,
                 attention_downsample_rate=2,
                 skip_first_layer_pos_embed=False,
                 attn_cls=QKVSeparateAttention):
        super(CrossAttentionBlock, self).__init__()
        self.self_attn = attn_cls(embed_dim=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn_token_to_image = attn_cls(
            embed_dim=embed_dim, num_heads=num_heads,
            downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = TwoLayerMLP(embed_dim, mlp_dim, act_layer)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.cross_attn_image_to_token = attn_cls(
            embed_dim=embed_dim, num_heads=num_heads,
            downsample_rate=attention_downsample_rate)
        self.norm4 = nn.LayerNorm(embed_dim)

        self.skip_first_layer_pos_embed = skip_first_layer_pos_embed

    def forward(self, query, key, query_pos_embed, key_pos_embed):
        if self.skip_first_layer_pos_embed:
            query = self.self_attn(query, query, query)
        else:
            query_with_pos_embed = query + query_pos_embed
            query = query + self.self_attn(
                query_with_pos_embed, query_with_pos_embed, query)
        query = self.norm1(query)

        query_with_pos_embed = query + query_pos_embed
        key_with_pos_embed = key + key_pos_embed
        query = query + self.cross_attn_token_to_image(
            query_with_pos_embed, key_with_pos_embed, key)
        query = self.norm2(query)

        query = query + self.mlp(query)
        query = self.norm3(query)

        query_with_pos_embed = query + query_pos_embed
        key_with_pos_embed = key + key_pos_embed
        key = key + self.cross_attn_image_to_token(
            key_with_pos_embed, query_with_pos_embed, query)
        key = self.norm4(key)
        return query, key


class CrossAttentionTransformer(nn.Module):

    def __init__(self,
                 depth,
                 embed_dim,
                 num_heads,
                 mlp_dim,
                 act_layer=nn.ReLU,
                 attention_downsample_rate=2,
                 attn_cls=CrossAttentionBlock
                 ) -> None:
        super(CrossAttentionTransformer, self).__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(attn_cls(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                act_layer=act_layer,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pos_embed=(i == 0)))

        self.final_attn_token_to_image = QKVSeparateAttention(
            embed_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embed_dim)

    def forward(self, image_embeds, image_pos_embeds, point_embeds):
        image_embeds = rearrange(image_embeds, 'b c h w -> b (h w) c')
        image_pos_embeds = rearrange(image_pos_embeds, 'b c h w -> b (h w) c')

        query, key = point_embeds, image_embeds
        for layer in self.layers:
            query, key = layer(query, key, point_embeds, image_pos_embeds)

        query_with_pos_embed = query + point_embeds
        key_with_pos_embed = key + image_pos_embeds
        query = query + self.final_attn_token_to_image(
            query_with_pos_embed, key_with_pos_embed, key)
        query = self.norm_final_attn(query)
        return query, key


@HEADS.register_module()
class SAMDecoder(BaseDecodeHead):

    def __init__(self,
                 in_dim,
                 attn_cls=CrossAttentionTransformer,
                 attn_cfg=dict(depth=2, mlp_dim=2048, num_heads=8),
                 num_multimask_outputs=3,
                 act_layer=nn.GELU,
                 iou_head_depth=3,
                 iou_head_hidden_dim=256,
                 num_classes=2,
                 align_corners=False,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 init_cfg=None):
        super(SAMDecoder, self).__init__(
            num_classes=num_classes,
            align_corners=align_corners,
            loss_decode=loss_decode,
            init_cfg=init_cfg)
        self.in_dim = in_dim
        self.transformer = attn_cls(embed_dim=in_dim, **attn_cfg)

        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        self.iou_token = nn.Embedding(1, in_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, in_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim // 4, 2, stride=2),
            LayerNorm2d(in_dim // 4),
            act_layer(),
            nn.ConvTranspose2d(in_dim // 4, in_dim // 8, 2, stride=2),
            act_layer())
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(in_dim, in_dim, in_dim // 8, 3)
            for _ in range(self.num_mask_tokens)])

        self.iou_prediction_head = MLP(
            in_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(self, inputs, mode='single mask'):
        if mode not in ['single mask', 'multiple masks',
                        'multiple masks with ious', 'best mask']:
            raise ValueError(f'Unknown mode {mode}, expected '
                             f'"single mask", "multiple masks", '
                             f'"multiple masks with ious" or "best mask"')
        masks, iou_pred = self.stem(inputs)
        if mode == 'single mask':
            return masks[:, :1]
        elif mode == 'multiple masks':
            return masks
        elif mode == 'multiple masks with ious':
            return masks, iou_pred
        elif mode == 'best mask':
            best_masks = F.one_hot(iou_pred.argmax(dim=1), masks.size(1))
            mask = masks[best_masks.bool()]
            return mask.view(masks.size(0), 1, *masks.shape[2:])
        else:
            raise NotImplementedError

    def stem(self, inputs):
        image_embeds = inputs['image_embeds']
        image_pos_embeds = inputs['image_pos_embeds']
        sparse_embeds = inputs['sparse_embeds']
        dense_embeds = inputs['dense_embeds']

        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = repeat(output_tokens, 'n d -> b n d',
                               b=sparse_embeds.size(0))
        tokens = torch.cat([output_tokens, sparse_embeds], dim=1)

        src = image_embeds
        src = src + dense_embeds
        pos_src = image_pos_embeds
        B, C, H, W = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0]
        mask_tokens_out = hs[:, 1:(self.num_mask_tokens + 1)]

        src = rearrange(src, 'b (h w) c -> b c h w', h=H, w=W)
        upscaled_embeds = self.output_upscaling(src)
        hyper_in_list = list()
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        H, W = upscaled_embeds.shape[-2:]
        masks = hyper_in @ rearrange(upscaled_embeds, 'b c h w -> b c (h w)')
        masks = rearrange(masks, 'b n (h w) -> b n h w', h=H, w=W)

        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


if __name__ == '__main__':
    model = SAMDecoder(
        in_dim=256,
        attn_cfg=dict(depth=2, mlp_dim=2048, num_heads=8),
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        align_corners=False,
        loss_decode=[dict(type='NormalizedFocalLoss', loss_weight=1.0),
                     dict(type='BinaryIoU')])

    count = 0
    for param in model.parameters():
        count += param.numel()
    if count > 1e6:
        count = count / 1e6
        print(f'Number of parameters: {count:.2f}M')
    elif count > 1e3:
        count = count / 1e3
        print(f'Number of parameters: {count:.2f}K')
    else:
        print(f'Number of parameters: {count}')

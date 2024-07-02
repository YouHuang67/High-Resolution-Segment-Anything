_base_ = ['../../_base_/optimizer_vit_adamw_160k.py',
          '../../datasets/train_colaug_coco_lvis_1024x1024.py',
          '../hrsam_plusplus.py']
find_unused_parameters = False

size = (1024, 1024)
data_preprocessor = dict(size=size)
model = dict(
    type='SimpleDistillation',
    image_embed_loader=dict(
        type='BaseEmbedLoader',
        embed_dir='data/embeds/colaug_coco_1024x1024_mmcs_sam_vit_huge'
    ),
    train_cfg=dict(target_size=size)
)
batch_size = 1
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size,
    sampler=dict(batch_size=batch_size)
)
optim_wrapper = dict(
    paramwise_cfg=dict(
        num_layers=12, ignore_keys=['mamba', 'pos_embed', 'patch_embed']
    )
)

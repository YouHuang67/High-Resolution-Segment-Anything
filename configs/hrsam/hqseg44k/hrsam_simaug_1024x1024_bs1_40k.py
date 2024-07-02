_base_ = ['../../_base_/optimizer_adamw_40k.py',
          '../../datasets_ext/train_simaug_hqseg44k_1024x1024.py',
          '../hrsam.py']
find_unused_parameters = True

size = (1024, 1024)
data_preprocessor = dict(size=size)
model = dict(

    remove_backbone=False,
    freeze_backbone=False,
    freeze_neck=False,
    freeze_decode_head=False,
    image_embed_loader=None,
    backbone=dict(drop_rate=0., attn_drop_rate=0., drop_path_rate=0.),

    init_cfg=dict(checkpoint='work_dirs/hrsam/coco_lvis/'
                             'simdist_hrsam_colaug_1024x1024_bs1_160k/'
                             'iter_160000.pth'),
    train_cfg=dict(
        max_num_clicks=20, gamma=0.6, sfc_inner_k=1.7,
        target_image_size=1024)
)

batch_size = 1
train_dataloader = dict(batch_size=batch_size, num_workers=batch_size)
optim_wrapper = dict(
    paramwise_cfg=dict(
        num_layers=12, ignore_keys=['mamba', 'pos_embed', 'patch_embed']
    )
)

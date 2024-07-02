data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=0,

    # uncomment this line to use 1024x1024 or other target size
    # size=(1024, 1024),

    size_divisor=None,
    test_cfg=dict(size_divisor=32)
)

model = dict(
    type='SimpleSegmentor',

    init_cfg=dict(type='Pretrained',
                  checkpoint='pretrain/mae_vit_base_sam_huge_dec.pth'),

    remove_backbone=False,
    freeze_backbone=False,
    freeze_neck=True,
    freeze_decode_head=True,

    backbone=dict(
        type='HRSAMViT',

        window_size=16,

        # mamba configs
        state_dim=32,
        extend_ratio=3,

        in_dim=3,
        img_size=224,
        patch_size=16,
        depth=12,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        use_checkpoint=False,
        out_indices=(2, 5, 8, 11),
        final_embed_dim=256),
    neck=dict(
        type='SAMPromptEncoder',
        embed_dim=256,
        image_embed_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_dim=16),
    decode_head=dict(
        type='SAMDecoder',
        in_dim=256,
        attn_cfg=dict(depth=2, mlp_dim=2048, num_heads=8),
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        align_corners=False,
        loss_decode=[dict(type='NormalizedFocalLoss', loss_weight=1.0),
                     dict(type='BinaryIoU')]),
    train_cfg=dict(
        max_num_clicks=20,
        gamma=0.6,
        sfc_inner_k=1.7,
        interact_params={'coco': dict(gamma=0.6, refine_gamma=0.6),
                         'lvis': dict(gamma=0.9, refine_gamma=0.35)},

        # uncomment this line to use 1024x1024 or other target size
        # target_image_size=1024

    ),
    test_cfg=dict(target_size=1024)
)
find_unused_parameters = True

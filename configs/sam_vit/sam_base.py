data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=0,
    test_cfg=dict(size_divisor=32)
)
model = dict(
    type='SimpleSegmentor',
    init_cfg=dict(type='Pretrained',
                  checkpoint='pretrain/sam_pretrain_vit_base_mmcls.pth'),
    backbone=dict(
        type='SAMWindowViT',
        img_size=1024,
        patch_size=16,
        in_dim=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_dim=256,
        qkv_bias=True,
        use_abs_pos_embed=True,
        use_rel_pos_embed=True,
        window_size=14,
        global_attn_indexes=(2, 5, 8, 11)),
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
    test_cfg=dict(target_size=1024)
)
find_unused_parameters = True

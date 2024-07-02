_base_ = 'runtime_40k.py'
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW',
                   lr=1e-4,
                   betas=(0.9, 0.999),
                   weight_decay=0.05)
)
param_scheduler = [dict(type='LinearLR',
                        start_factor=1e-6,
                        by_epoch=False, begin=0, end=1500),
                   dict(type='PolyLR',
                        eta_min=0.0,
                        power=1.0,
                        begin=1500,
                        end=40000,
                        by_epoch=False)]

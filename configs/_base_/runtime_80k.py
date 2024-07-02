_base_ = 'runtime_40k.py'
train_cfg = dict(max_iters=80000, val_interval=20000)
default_hooks = dict(checkpoint=dict(interval=20000))

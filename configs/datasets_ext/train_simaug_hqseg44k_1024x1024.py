target_size = 1024
crop_size = (target_size, target_size)

# HQSeg44K settings
dataset_type = 'ExtHQSeg44kTrainDataset'
data_root = 'data/sam-hq'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadExtHQSeg44kTrainAnnotations'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='InterSegPackSegInputs')
]
dataset = dict(type=dataset_type, data_root=data_root, pipeline=train_pipeline)

batch_size = 4
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler'),
    dataset=dataset
)

crop_size = (1024, 1024)

# LVIS settings
dataset_type = 'LVISDataset'
data_root = 'data/lvis'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadLVISAnnotations'),
    dict(type='ObjectSampler',
         max_num_merged_objects=1,
         min_area_ratio=0.0),
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

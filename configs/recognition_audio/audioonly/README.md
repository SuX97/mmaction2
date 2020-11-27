# AudioOnly

## Introduction
```
@article{xiao2020audiovisual,
  title={Audiovisual SlowFast Networks for Video Recognition},
  author={Xiao, Fanyi and Lee, Yong Jae and Grauman, Kristen and Malik, Jitendra and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2001.08740},
  year={2020}
}
```

## Model Zoo

### Kinetics-400

|config | n_fft | gpus | backbone |pretrain| top1 acc/delta| top5 acc/delta | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[audioonly_r50_64x1x1_100e_kinetics400_audio_feature](/configs/recognition_audio/audioonly/audioonly_r50_64x1x1_100e_kinetics400_audio_feature.py)|1024|8x2| ResNet50 | None |21.88|39.65|x|24613|[ckpt](https://download.openmmlab.com/mmaction/recognition_audio/audioonly/audioonly_r50_64x1x1_100e_kinetics400_audio_feature/audioonly_r50_64x1x1_100e_kinetics400_audio_feature_20201127-9d591fed.pth)|[log](https://download.openmmlab.com/mmaction/recognition_audio/audioonly/audioonly_r50_64x1x1_100e_kinetics400_audio_feature/20201126_161659.log)|[json](https://download.openmmlab.com/mmaction/recognition_audio/audioonly/audioonly_r50_64x1x1_100e_kinetics400_audio_feature/20201126_161659.log.json)|

Notes:

1. The **gpus** indicates the number of gpus we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train an AudioOnly model on Kinetics-400 audio dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/audio_recognition/audioonly_r50_64x1x1_100e_kinetics400_audio_feature.py \
    --work-dir work_dirs/audioonly_r50_64x1x1_100e_kinetics400_audio_feature \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test an AudioOnly model on Kinetics-400 audio dataset and dump the result to a json file.
```shell
python tools/test.py configs/audio_recognition/audioonly_r50_64x1x1_100e_kinetics400_audio_feature.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
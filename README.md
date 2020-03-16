# FeatureFlow

A state-of-the-art Video Frame Interpolation Method using deep semantic flows blending.

CVPR 2020: FeatureFlow: Robust Video Interpolation via Structure-to-texture Generation

* PyTorch 1.1 or higher
* mmdet 1.0rc (from https://github.com/open-mmlab/mmdetection.git)
* visdom *

## Video demos

Click the picture to Download one of them or click [Here](https://drive.google.com/open?id=1QUYoplBNjaWXJZPO90NiwQwqQz7yF7TX) to download all.
[<img width="320" height="180" src="https://github.com/CM-BF/FeatureFlow/blob/master/videos/youtube.png"/>](https://github.com/CM-BF/FeatureFlow/blob/master/videos/youtube.mp4 "video1")
[<img width="320" height="180" src="https://github.com/CM-BF/storage/tree/master/videos/check_all.png"/>](https://github.com/CM-BF/storage/tree/master/videos/check_all.mp4 "video2")
[<img width="320" height="180" src="https://github.com/CM-BF/storage/tree/master/videos/tianqi_all.png"/>](https://github.com/CM-BF/storage/tree/master/videos/tianqi_all.mp4 "video3")
[<img width="320" height="180" src="https://github.com/CM-BF/storage/tree/master/videos/video.png"/>](https://github.com/CM-BF/storage/tree/master/videos/video.mp4 "video4")
[<img width="320" height="180" src="https://github.com/CM-BF/storage/tree/master/videos/shana.png"/>](https://github.com/CM-BF/storage/tree/master/videos/shana.mp4 "video5")


## Steps
* clone this repo
* git clone https://github.com/open-mmlab/mmdetection.git
* install mmdetection: please follow the guidence in its github
```bash
$ cd mmdetection
$ pip install -r requirements/build.txt
$ pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
$ pip install -v -e .  # or "python setup.py develop"
$ pip list | grep mmdet
```
* Download [test set](http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip)
```bash
$ unzip vimeo_interp_test.zip
$ cd vimeo_interp_test
$ mkdir sequences
$ cp target/* sequences/ -r
$ cp input/* sequences/ -r
```
* Download BDCN's pre-trained model:bdcn_pretrained_on_bsds500.pth to ./model/bdcn/final-model/
```
$ pip install scikit-image visdom tqdm prefetch-generator
```

## Pre-trained Model

[Google Drive](https://drive.google.com/open?id=1S8C0chFV6Bip6W9lJdZkog0T3xiNxbEx)

Place FeFlow.ckpt to ./checkpoints/.

## Download Results

[Google Drive](https://drive.google.com/open?id=1OtrExUiyIBJe0D6_ZwDfztqJBqji4lmt)

## Evaluation
```bash
$ CUDA_VISIBLE_DEVICES=0 python eval_Vimeo90K.py --checkpoint ./checkpoints/FeFlow.ckpt --dataset_root ~/datasets/videos/vimeo_interp_test --visdom_env test --vimeo90k --imgpath ./results/
```

## Video processing
```bash
$ CUDA_VISIBLE_DEVICES=0 python video_process.py --checkpoint checkpoints/FeFlow.ckpt --video_name ./youvideo.mp4  --fix_range
```

## Citation
```
@InProceedings{FeatureFlow,
author = {Gui, Shurui and Wang, Chaoyue and Chen, Qihua and Tao, Dacheng},
title = {FeatureFlow: Robust Video Interpolation via Structure-to-texture Generation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Contact
[Shurui Gui](mailto:citrinegui@gmail.com); [Chaoyue Wang](mailto:chaoyue.wang@sydney.edu.au)

## License
See [MIT License](https://github.com/CM-BF/FeatureFlow/blob/master/LICENSE)

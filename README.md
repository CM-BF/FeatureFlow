# FeatureFlow

A state-of-the-art Video Frame Interpolation Method using deep semantic flows blending.

* PyTorch 1.1 or higher
* mmdet 1.0rc (from https://github.com/open-mmlab/mmdetection.git)
* visdom

# Step
* clone this repo
* git clone https://github.com/open-mmlab/mmdetection.git
* install mmdetection: please follow the guidence in github
```bash
cd mmdetection
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .  # or "python setup.py develop"
pip list | grep mmdet
```
* Download http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip
```bash
unzip it
cd vimeo_interp_test
mkdir sequences
cp target/* sequences/ -r
cp input/* sequences/ -r
```
* Download BDCN's pre-trained model:bdcn_pretrained_on_bsds500.pth to ./model/bdcn/final-model/
`pip install scikit-image visdom tqdm prefetch-generator`
run with:
`CUDA_VISIBLE_DEVICES=0 python eval_Vimeo90K.py --checkpoint ./checkpoints/SeDraw29.ckpt --dataset_root ~/datasets/videos/vimeo_interp_test --visdom_env test --vimeo90k --imgpath ./results/`

# video processing
```bash
CUDA_VISIBLE_DEVICES=0 python video_process.py --checkpoint checkpoints/SeDraw29.ckpt --video_name ./youvideo.mp4  --fix_range
```



# SemanticFlow

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
* if you see the "mmdet           1.0+93bed07" then you succeed installing mmdet
* Download http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip
* unzip it
* cd vimeo_interp_test
* mkdir sequences
* cp target/* sequences/ -r
* cp input/* sequences/ -r
* Download BDCN's pre-trained model:bdcn_pretrained_on_bsds500.pth to ./model/bdcn/final-model/
* pip install scikit-image visdom tqdm prefetch-generator
run with:
`CUDA_VISIBLE_DEVICES=0 python eval_Vimeo90K.py --checkpoint ./checkpoints/SeDraw29.ckpt --dataset_root ~/datasets/videos/vimeo_interp_test --visdom_env test --vimeo90k --imgpath ./results/`

# video processing
```python
CUDA_VISIBLE_DEVICES=0 python video_process.py --checkpoint checkpoints/SeDraw29.ckpt --video_name ./youtube-35d8xb6ymp4_Ocw3.mp4  --fix_range
```



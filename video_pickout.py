# SeDraw
import argparse
import os
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import src.pure_network as layers
from tqdm import tqdm
import numpy as np
import math
import models.bdcn.bdcn as bdcn

# For parsing commandline arguments
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True, help='the path to the input video.')
args = parser.parse_args()


def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):


    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')
def main():
    # initial

    if not os.path.isfile(args.video):
        print('video not exist!')
    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS) #// 2
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    video_writer = [cv2.VideoWriter(args.video[:-4] + '_p1.mp4',
                                   fourcc,
                                   fps,
                                   size), cv2.VideoWriter(args.video[:-4] + '_p2.mp4',
                                   fourcc,
                                   fps,
                                   size)]

    s_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print('out')
            break
        video_writer[s_count].write(frame)
        s_count = (s_count + 1) % 2

    video_writer[0].release()
    video_writer[1].release()


main()



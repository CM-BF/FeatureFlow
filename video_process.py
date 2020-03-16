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
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
parser.add_argument('--feature_level', type=int, default=3, help='Using feature_level=? in GEN, Default:3')
parser.add_argument('--bdcn_model', type=str, default='./models/bdcn/final-model/bdcn_pretrained_on_bsds500.pth')
parser.add_argument('--DE_pretrained', action='store_true', help='using this flag if training the model from pretrained parameters.')
parser.add_argument('--DE_ckpt', type=str, help='path to DE checkpoint')
parser.add_argument('--video_name', type=str, required=True, help='the path the  video.')
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--fix_range', action='store_true', help="it won't change the fps without this flag.")
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



bdcn = bdcn.BDCN()
bdcn.cuda()
structure_gen = layers.StructureGen(feature_level=args.feature_level)
structure_gen.cuda()
detail_enhance = layers.DetailEnhance()
detail_enhance.cuda()


# Channel wise mean calculated on adobe240-fps training dataset
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
normalize = transforms.Normalize(mean=mean,
                                 std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

negmean = [-1 for x in mean]
restd = [2, 2, 2]
revNormalize = transforms.Normalize(mean=negmean, std=restd)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])


def ToImage(frame0, frame1):

    with torch.no_grad():

        img0 = frame0.cuda()
        img1 = frame1.cuda()

        img0_e = torch.cat([img0, torch.tanh(bdcn(img0)[0])], dim=1)
        img1_e = torch.cat([img1, torch.tanh(bdcn(img1)[0])], dim=1)
        ref_imgt, _ = structure_gen((img0_e, img1_e))
        imgt = detail_enhance((img0, img1, ref_imgt))
        # imgt = detail_enhance((img0, img1, imgt))
        imgt = torch.clamp(imgt, max=1., min=-1.)

    return imgt


def main():
    # initial

    bdcn.load_state_dict(torch.load('%s' % (args.bdcn_model)))
    dict1 = torch.load(args.checkpoint)
    structure_gen.load_state_dict(dict1['state_dictGEN'], strict=False)
    detail_enhance.load_state_dict(dict1['state_dictDE'], strict=False)

    bdcn.eval()
    structure_gen.eval()
    detail_enhance.eval()

    if not os.path.isfile(args.video_name):
        print('video not exist!')
    video = cv2.VideoCapture(args.video_name)
    if args.fix_range:
        fps = video.get(cv2.CAP_PROP_FPS) * 2
    else:
        # fps = video.get(cv2.CAP_PROP_FPS)
        fps = 25
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    video_writer = cv2.VideoWriter(args.video_name[:-4] + '_Sedraw.mp4',
                                   fourcc,
                                   fps,
                                   size)

    flag = True
    frame_group = []
    while video.isOpened():
        for i in range(args.batchsize):
            ret, frame = video.read()
            if ret:
                frame = torch.FloatTensor(frame[:, :, ::-1].transpose(2, 0, 1).copy()) / 255
                frame = normalize(frame).unsqueeze(0)
                frame_group += [frame]
            else:
                break
        if len(frame_group) <= 1:
            break
        first = torch.cat(frame_group[:-1], dim=0)
        second = torch.cat(frame_group[1:], dim=0)

        middle_frame = ToImage(first, second)

        if flag:
            for i in range(first.shape[0]):
                first_np = first[i].cpu().numpy()
                first_png = np.uint8(((first_np + 1.0) / 2.0).transpose(1, 2, 0)[:, :, ::-1] * 255)
                middle_frame_np = middle_frame[i].cpu().numpy()
                middle_frame_png = np.uint8(((middle_frame_np + 1.0) / 2.0).transpose(1, 2, 0)[:, :, ::-1] * 255)
                video_writer.write(first_png)
                video_writer.write(middle_frame_png)
            second_np = second[-1].cpu().numpy()
            second_png = np.uint8(((second_np + 1.0) / 2.0).transpose(1, 2, 0)[:, :, ::-1] * 255)
            video_writer.write(second_png)
            frame_group = [second[-1].unsqueeze(0)]
            flag = False
        else:
            for i in range(second.shape[0]):
                middle_frame_np = middle_frame[i].cpu().numpy()
                middle_frame_png = np.uint8(((middle_frame_np + 1.0) / 2.0).transpose(1, 2, 0)[:, :, ::-1] * 255)
                second_np = second[i].cpu().numpy()
                second_png = np.uint8(((second_np + 1.0) / 2.0).transpose(1, 2, 0)[:, :, ::-1] * 255)
                video_writer.write(middle_frame_png)
                video_writer.write(second_png)
            frame_group = [second[-1].unsqueeze(0)]

    video_writer.release()


main()



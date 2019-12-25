# SeDraw
import argparse
import os
import torch
import cv2
import torchvision.transforms as transforms
from skimage.measure import compare_psnr
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
parser.add_argument('--bdcn_model', type=str, default='/home/visiting/Projects/citrine/SeDraw/models/bdcn/final-model/bdcn_pretrained_on_bsds500.pth')
parser.add_argument('--DE_pretrained', action='store_true', help='using this flag if training the model from pretrained parameters.')
parser.add_argument('--DE_ckpt', type=str, help='path to DE checkpoint')
parser.add_argument('--imgpath', type=str, required=True)
parser.add_argument('--first', type=str, required=True)
parser.add_argument('--second', type=str, required=True)
parser.add_argument('--gt', type=str, required=True)
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

    IE = 0
    PSNR = 0
    count = 0
    for folder in tqdm(os.listdir(args.imgpath)):
        triple_path = os.path.join(args.imgpath, folder)
        if not (os.path.isdir(triple_path)):
            continue
        X0 = transform(_pil_loader('%s/%s' % (triple_path, args.first))).unsqueeze(0)
        X1 = transform(_pil_loader('%s/%s' % (triple_path, args.second))).unsqueeze(0)

        assert (X0.size(2) == X1.size(2))
        assert (X0.size(3) == X1.size(3))

        intWidth = X0.size(3)
        intHeight = X0.size(2)
        channel = X0.size(1)
        if not channel == 3:
            print('Not RGB image')
            continue
        count += 1

        # if intWidth != ((intWidth >> 4) << 4):
        #     intWidth_pad = (((intWidth >> 4) + 1) << 4)  # more than necessary
        #     intPaddingLeft = int((intWidth_pad - intWidth) / 2)
        #     intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        # else:
        #     intWidth_pad = intWidth
        #     intPaddingLeft = 0
        #     intPaddingRight = 0
        #
        # if intHeight != ((intHeight >> 4) << 4):
        #     intHeight_pad = (((intHeight >> 4) + 1) << 4)  # more than necessary
        #     intPaddingTop = int((intHeight_pad - intHeight) / 2)
        #     intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        # else:
        #     intHeight_pad = intHeight
        #     intPaddingTop = 0
        #     intPaddingBottom = 0
        #
        # pader = torch.nn.ReflectionPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

        # first, second = pader(X0), pader(X1)
        first, second = X0, X1
        imgt = ToImage(first, second)

        imgt_np = imgt.squeeze(0).cpu().numpy()#[:, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]
        imgt_png = np.uint8(((imgt_np + 1.0) / 2.0).transpose(1, 2, 0)[:, :, ::-1] * 255)
        if not os.path.isdir(triple_path):
            os.system('mkdir -p %s' % triple_path)
        cv2.imwrite(triple_path + '/SeDraw.png', imgt_png)

        rec_rgb = np.array(_pil_loader('%s/%s' % (triple_path, 'SeDraw.png')))
        gt_rgb = np.array(_pil_loader('%s/%s' % (triple_path, args.gt)))

        diff_rgb = rec_rgb - gt_rgb
        avg_interp_error_abs = np.sqrt(np.mean(diff_rgb ** 2))

        mse = np.mean((diff_rgb) ** 2)

        PIXEL_MAX = 255.0
        psnr = compare_psnr(gt_rgb, rec_rgb, 255)
        print(folder, psnr)

        IE += avg_interp_error_abs
        PSNR += psnr

        # print(triple_path, ': IE/PSNR:', avg_interp_error_abs, psnr)

    IE = IE / count
    PSNR = PSNR / count
    print('Average IE/PSNR:', IE, PSNR)

main()



# SeDraw
import argparse
from skimage.measure import compare_psnr, compare_ssim
import numpy as np
import os
import torch
import cv2
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import src.dataloader as dataloader
import src.layers as layers
from math import log10
import datetime
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import visdom
from utils.visualize import feature_transform
import numpy as np
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
parser.add_argument("--dataset_root", type=str, required=True,
                    help='path to dataset folder containing train-test-validation folders')
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
parser.add_argument("--test_batch_size", type=int, default=6, help='batch size for training. Default: 6.')
parser.add_argument('--visdom_env', type=str, default='SeDraw_0', help='Environment for visdom show')
parser.add_argument('--vimeo90k', action='store_true', help='use this flag if using Vimeo-90K dataset')
parser.add_argument('--feature_level', type=int, default=3, help='Using feature_level=? in GEN, Default:3')
parser.add_argument('--bdcn_model', default='/home/visiting/Projects/citrine/SeDraw/models/bdcn/final-model/bdcn_pretrained_on_bsds500.pth')
parser.add_argument('--DE_pretrained', action='store_true', help='using this flag if training the model from pretrained parameters.')
parser.add_argument('--DE_ckpt', type=str, help='path to DE checkpoint')
parser.add_argument('--imgpath', type=str, required=True)
args = parser.parse_args()



# --For visualizing loss and interpolated frames--


# Visdom for real-time visualizing
vis = visdom.Visdom(env=args.visdom_env, port=8098)

# device
device_count = torch.cuda.device_count()

# --Initialize network--
bdcn = bdcn.BDCN()
bdcn.cuda()
structure_gen = layers.StructureGen(feature_level=args.feature_level)
structure_gen.cuda()
detail_enhance = layers.DetailEnhance()
detail_enhance.cuda()


# --Load Datasets--


# Channel wise mean calculated on adobe240-fps training dataset
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
normalize = transforms.Normalize(mean=mean,
                                 std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

if args.vimeo90k:
    testset = dataloader.SeDraw_vimeo90k(root=args.dataset_root, transform=transform,
                                      randomCropSize=(448, 256), train=False, test=True)
else:
    testset = dataloader.SeDraw(root=args.dataset_root + '/validation', transform=transform,
                                      randomCropSize=(448, 256), train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size * device_count, shuffle=False)

print(testset)

# --Create transform to display image from tensor--


negmean = [-1 for x in mean]
restd = [2, 2, 2]
revNormalize = transforms.Normalize(mean=negmean, std=restd)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])


# --Utils--

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



# --Validation function--
#


def validate():
    # For details see training.
    psnr = 0
    ie = 0
    tloss = 0

    with torch.no_grad():
        for testIndex, testData in tqdm(enumerate(testloader, 0)):
            frame0, frameT, frame1 = testData

            img0 = frame0.cuda()
            img1 = frame1.cuda()
            IFrame = frameT.cuda()

            img0_e = torch.cat([img0, torch.tanh(bdcn(img0)[0])], dim=1)
            img1_e = torch.cat([img1, torch.tanh(bdcn(img1)[0])], dim=1)
            IFrame_e = torch.cat([IFrame, torch.tanh(bdcn(IFrame)[0])], dim=1)
            _, _, ref_imgt = structure_gen((img0_e, img1_e, IFrame_e))
            loss, MSE_val, IE, imgt = detail_enhance((img0, img1, IFrame, ref_imgt))
            imgt = torch.clamp(imgt, max=1., min=-1.)
            IFrame_np = IFrame.squeeze(0).cpu().numpy()
            imgt_np = imgt.squeeze(0).cpu().numpy()
            imgt_png = np.uint8(((imgt_np + 1.0) / 2.0).transpose(1, 2, 0)[:, :, ::-1] * 255)
            IFrame_png = np.uint8(((IFrame_np + 1.0) /2.0).transpose(1, 2, 0)[:, :, ::-1] * 255)
            imgpath = args.imgpath + '/' + str(testIndex)
            if not os.path.isdir(imgpath):
                os.system('mkdir -p %s' % imgpath)
            cv2.imwrite(imgpath + '/imgt.png', imgt_png)
            cv2.imwrite(imgpath + '/IFrame.png', IFrame_png)

            PSNR = compare_psnr(IFrame_np, imgt_np, data_range=2)
            print('PSNR:', PSNR)

            loss = torch.mean(loss)
            MSE_val = torch.mean(MSE_val)

            if testIndex % 100 == 99:
                vImg = torch.cat([revNormalize(frame0[0]).unsqueeze(0), revNormalize(frame1[0]).unsqueeze(0),
                                  revNormalize(imgt.cpu()[0]).unsqueeze(0), revNormalize(frameT[0]).unsqueeze(0),
                                  revNormalize(ref_imgt.cpu()[0]).unsqueeze(0)],
                                 dim=0)


                vImg = torch.clamp(vImg, max=1., min=0)
                vis.images(vImg, win='vImage', env=args.visdom_env, nrow=2, opts={'title': 'visual_image'})

            # psnr
            tloss += loss.item()

            psnr += PSNR
            ie += IE

    return (psnr / len(testloader)), (tloss / len(testloader)), MSE_val, (ie / len(testloader))


# --Initialization--

bdcn.load_state_dict(torch.load('%s' % (args.bdcn_model)))

dict1 = torch.load(args.checkpoint)
structure_gen.load_state_dict(dict1['state_dictGEN'])
detail_enhance.load_state_dict(dict1['state_dictDE'])

start = time.time()

bdcn.eval()
structure_gen.eval()
detail_enhance.eval()
psnr, vLoss, MSE_val, ie = validate()
end = time.time()

print(" Loss: %0.6f  TestExecTime: %0.1f  ValPSNR: %0.4f ValIE: %0.4f" % (
    vLoss, end - start, psnr, ie))


# FeatureFlow

import argparse
import torch
import os
from skimage.measure import compare_ssim
import torchvision
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import src.dataloader as dataloader
import src.layers as layers
from math import log10
import datetime
from tqdm import tqdm
import visdom
import models.bdcn.bdcn as bdcn
from utils.visualize import feature_transform, edge_transform
import numpy as np

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
                    help='[important]path to dataset folder containing train-test-validation folders')
parser.add_argument("--checkpoint_dir", type=str, required=True, help='path to folder for saving checkpoints')
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
parser.add_argument("--train_continue", action='store_true',
                    help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
parser.add_argument("--epochs", type=int, default=200, help='[You can terminate program as our paper described]number of epochs to train. Default: 200.')
parser.add_argument("--train_batch_size", type=int, default=6, help='batch size for training. Default: 6.')
parser.add_argument("--validation_batch_size", type=int, default=10, help='[It depends]batch size for validation. Default: 10.')
parser.add_argument("--init_learning_rate", type=float, default=0.0001,
                    help='[Require]set initial learning rate. Default: 0.0001.')
parser.add_argument("--milestones", type=int, required=True, nargs='+',
                    help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. '
                         'Default: [100, 150]')
parser.add_argument("--progress_iter", type=int, default=100,
                    help='[Default]frequency of reporting progress and validation. N: after every N iterations. Default: 100.')
parser.add_argument("--checkpoint_epoch", type=int, default=5,
                    help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of '
                         'size 151 MB.Default: 5.')
parser.add_argument('--visdom_env', type=str, default='FeFlow_0', help='Environment for visdom show')
parser.add_argument('--vimeo90k', action='store_true', help='[Must be used]use this flag if using Vimeo-90K dataset')
parser.add_argument('--GEN_DE', type=str2bool, nargs='?', const=True, default=False, help='[Important]True: train generator, False: train DE')
parser.add_argument('--test', action='store_true', help='if debug network by using 1 image, Default:False')
parser.add_argument('--fp16', action='store_true', help='[discarded]using apex for fp16')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--feature_level', type=int, default=3, help='[Do not change it if you do not know what it means]Using feature_level=? in GEN, Default:3')
parser.add_argument('--bdcn_model', default='./models/bdcn/final-model/bdcn_pretrained_on_bsds500.pth')
parser.add_argument('--DE_pretrained', action='store_true', help='[You do not need to use it for training.]using this flag if training the model from pretrained parameters.')
parser.add_argument('--DE_ckpt', type=str, help='path to DE checkpoint')
parser.add_argument('--final', action='store_true', help='[Do not use it.]True: train all together')
args = parser.parse_args()

# for saving checkpoint
if not os.path.isdir(args.checkpoint_dir):
    os.system('mkdir -p %s' % args.checkpoint_dir)


# --For visualizing loss and interpolated frames--


# Visdom for real-time visualizing
vis = visdom.Visdom(env=args.visdom_env, port=8098)

# device
device_count = torch.cuda.device_count()

# --Initialize network--
bdcn = bdcn.BDCN()
bdcn.cuda()
structure_gen = nn.DataParallel(layers.StructureGen(feature_level=args.feature_level))
structure_gen.cuda()
detail_enhance = nn.DataParallel(layers.DetailEnhance())
detail_enhance.cuda()
if args.final:
    detail_enhance_last = nn.DataParallel(layers.DetailEnhance_last())
    detail_enhance_last.cuda()


# --Load Datasets--


# Channel wise mean calculated on adobe240-fps training dataset
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
normalize = transforms.Normalize(mean=mean,
                                 std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

if args.vimeo90k:
    trainset = dataloader.FeFlow_vimeo90k(root=args.dataset_root, transform=transform, train=True)
else:
    trainset = dataloader.FeFlow(root=args.dataset_root + '/train', transform=transform, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size * device_count, shuffle=True)

if args.vimeo90k:
    validationset = dataloader.FeFlow_vimeo90k(root=args.dataset_root, transform=transform,
                                      randomCropSize=(448, 256), train=False)
else:
    validationset = dataloader.FeFlow(root=args.dataset_root + '/validation', transform=transform,
                                      randomCropSize=(448, 256), train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.validation_batch_size * device_count, shuffle=False)

print(trainset, validationset)

# --Create transform to display image from tensor--


negmean = [-1 for x in mean]
restd = [2, 2, 2]
revNormalize = transforms.Normalize(mean=negmean, std=restd)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])


# --Utils--

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# --Loss and Optimizer--
if args.final:
    for param in structure_gen.parameters(True):
        param.requires_grad = False
    for param in detail_enhance.parameters(True):
        param.requires_grad = False
    optimizer = optim.Adam(detail_enhance_last.parameters(), lr=args.init_learning_rate)
else:
    if args.GEN_DE:
        for param in detail_enhance.parameters(True):
            param.requires_grad = False
        optimizer = optim.Adam(structure_gen.parameters(), lr=args.init_learning_rate)
    else:
        for param in structure_gen.parameters(True):
            param.requires_grad = False
        optimizer = optim.Adam(detail_enhance.parameters(), lr=args.init_learning_rate)
for param in bdcn.parameters(True):
    param.requires_grad = False
# scheduler to decrease learning rate by a factor of 10 at milestones.
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.25)


# --Validation function--
#


def validate():
    # For details see training.
    psnr = 0
    ssim = 0
    tloss = 0
    flag = 1

    with torch.no_grad():
        for validationIndex, validationData in enumerate(validationloader, 0):
            frame0, frameT, frame1 = validationData

            img0 = frame0.cuda()
            img1 = frame1.cuda()
            IFrame = frameT.cuda()

            img0_e = torch.cat([img0, torch.tanh(bdcn(img0)[0])], dim=1)
            img1_e = torch.cat([img1, torch.tanh(bdcn(img1)[0])], dim=1)
            IFrame_e = torch.cat([IFrame, torch.tanh(bdcn(IFrame)[0])], dim=1)

            if args.final:
                _, _, ref_imgt = structure_gen((img0_e, img1_e, IFrame_e))
                _, _, _, ref_imgt2 = detail_enhance((img0, img1, IFrame, ref_imgt))
                loss, MSE_val, imgt = detail_enhance_last((img0, img1, IFrame, ref_imgt2))
                SSIM = compare_ssim(IFrame.reshape(1, -1, 256, 448).squeeze(0).cpu().numpy().transpose(1, 2, 0),
                                    imgt.reshape(1, -1, 256, 448).squeeze(0).cpu().numpy().transpose(1, 2, 0),
                                    data_range=2, multichannel=True)
            else:
                if args.GEN_DE:
                    loss, MSE_val, ref_imgt = structure_gen((img0_e, img1_e, IFrame_e))
                    SSIM = compare_ssim(IFrame.reshape(1, -1, 256, 448).squeeze(0).cpu().numpy().transpose(1, 2, 0),
                                        ref_imgt.reshape(1, -1, 256, 448).squeeze(0).cpu().numpy().transpose(1, 2, 0),
                                        data_range=2, multichannel=True)
                else:
                    _, _, ref_imgt = structure_gen((img0_e, img1_e, IFrame_e))
                    loss, MSE_val, _, imgt = detail_enhance((img0, img1, IFrame, ref_imgt))
                    SSIM = compare_ssim(IFrame.reshape(1, -1, 256, 448).squeeze(0).cpu().numpy().transpose(1, 2, 0),
                                        imgt.reshape(1, -1, 256, 448).squeeze(0).cpu().numpy().transpose(1, 2, 0), data_range=2, multichannel=True)
            loss = torch.mean(loss)
            MSE_val = torch.mean(MSE_val)

            if (flag):
                if args.final:
                    retImg = torch.cat([revNormalize(frame0[0]).unsqueeze(0), revNormalize(frame1[0]).unsqueeze(0),
                                        revNormalize(imgt.cpu()[0]).unsqueeze(0), revNormalize(frameT[0]).unsqueeze(0),
                                        revNormalize(ref_imgt2.cpu()[0]).unsqueeze(0), revNormalize(ref_imgt.cpu()[0]).unsqueeze(0)],
                                       dim=0)
                else:
                    if args.GEN_DE:
                        retImg = torch.cat([revNormalize(frame0[0]).unsqueeze(0), revNormalize(frame1[0]).unsqueeze(0),
                                            revNormalize(ref_imgt.cpu()[0]).unsqueeze(0), revNormalize(frameT[0]).unsqueeze(0)],
                                           dim=0)
                    else:
                        retImg = torch.cat([revNormalize(frame0[0]).unsqueeze(0), revNormalize(frame1[0]).unsqueeze(0),
                                            revNormalize(imgt.cpu()[0]).unsqueeze(0), revNormalize(frameT[0]).unsqueeze(0),
                                            revNormalize(ref_imgt.cpu()[0]).unsqueeze(0)],
                                           dim=0)
                flag = 0

            # psnr
            tloss += loss.item()

            psnr += (10 * log10(4 / MSE_val.item()))
            ssim += SSIM

    return (psnr / len(validationloader)), (tloss / len(validationloader)), retImg, MSE_val, (ssim / len(validationloader))


# --Initialization--
bdcn.load_state_dict(torch.load('%s' % (args.bdcn_model)))

if args.train_continue:
    dict1 = torch.load(args.checkpoint)
    if args.final:
        structure_gen.module.load_state_dict(dict1['state_dictGEN'], strict=False)
        detail_enhance.module.load_state_dict(dict1['state_dictDE'])
        detail_enhance_last.module.load_state_dict(dict1['state_dictDE_last'])
    else:
        if args.GEN_DE:
            structure_gen.module.load_state_dict(dict1['state_dictGEN'])
        else:
            structure_gen.module.load_state_dict(dict1['state_dictGEN'], strict=False)
            detail_enhance.module.load_state_dict(dict1['state_dictDE'])
    for _ in range(dict1['epoch']):
        scheduler.step()
else:
    if args.final:
        dict1 = torch.load(args.checkpoint)
        structure_gen.module.load_state_dict(dict1['state_dictGEN'])
        detail_enhance.module.load_state_dict(dict1['state_dictDE'])
    else:
        if not args.GEN_DE:
            dict1 = torch.load(args.checkpoint)
            structure_gen.module.load_state_dict(dict1['state_dictGEN'])
            if args.DE_pretrained:
                dict2 = torch.load(args.DE_ckpt)
                detail_enhance.module.load_state_dict(dict2['state_dictDE'])
    dict1 = {'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}
# --Training--

start = time.time()
cLoss = dict1['loss']
valLoss = dict1['valLoss']
valPSNR = dict1['valPSNR']
checkpoint_counter = int((dict1['epoch'] + 1) / args.checkpoint_epoch)

if args.final:
    structure_gen.eval()
    detail_enhance.eval()
    detail_enhance_last.train()
else:
    if args.GEN_DE:
        structure_gen.train()
    else:
        structure_gen.eval()
        detail_enhance.train()
bdcn.eval()

# --Main training loop--
for epoch in range(dict1['epoch'] + 1, args.epochs):
    print("Epoch: ", epoch)

    # Append and reset
    cLoss.append([])
    valLoss.append([])
    valPSNR.append([])
    iLoss = 0

    # Increment scheduler count
    scheduler.step()

    if args.test:
        iter_data = iter(trainloader)
        test_batch = iter_data.next()
        for i in range(3):
            test_batch = iter_data.next()
    for trainIndex, trainData in tqdm(enumerate(trainloader, 0)):

        # Getting the input and the target from the training set
        start_time = time.time()
        frame0, frameT, frame1 = trainData
        if args.test:
            """
            just for 1 batch test
            """
            frame0, frameT, frame1 = test_batch

        img0 = frame0.cuda()
        img1 = frame1.cuda()
        IFrame = frameT.cuda()

        img0_e = torch.cat([img0, torch.tanh(bdcn(img0)[0])], dim=1)
        img1_e = torch.cat([img1, torch.tanh(bdcn(img1)[0])], dim=1)
        IFrame_e = torch.cat([IFrame, torch.tanh(bdcn(IFrame)[0])], dim=1)

        if args.final:
            _, _, ref_imgt = structure_gen((img0_e, img1_e, IFrame_e))
            _, _, _, ref_imgt2 = detail_enhance((img0, img1, IFrame, ref_imgt))
            loss, MSE_val, imgt = detail_enhance_last((img0, img1, IFrame, ref_imgt2))
        else:
            if args.GEN_DE:
                loss, _, ref_imgt = structure_gen((img0_e, img1_e, IFrame_e))
            else:
                _, _, ref_imgt = structure_gen((img0_e, img1_e, IFrame_e))
                loss, _, _, imgt = detail_enhance((img0, img1, IFrame, ref_imgt))
            # print(torch.max(torch.abs(ref_imgt - IFrame)))

        # loss, _ = loss_function(img0, img1, IFrame, *output)
        loss = torch.mean(loss)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()
        iLoss += loss.item()

        if trainIndex % 5 == 4:
            if args.final:
                vImg = torch.cat([revNormalize(frame0[0]).unsqueeze(0), revNormalize(frame1[0]).unsqueeze(0),
                                  revNormalize(imgt.cpu()[0]).unsqueeze(0), revNormalize(frameT[0]).unsqueeze(0),
                                  revNormalize(ref_imgt2.cpu()[0]).unsqueeze(0), revNormalize(ref_imgt.cpu()[0]).unsqueeze(0)],
                                 dim=0)
            else:
                if args.GEN_DE:
                    vImg = torch.cat([revNormalize(frame0[0]).unsqueeze(0), revNormalize(frame1[0]).unsqueeze(0),
                                      revNormalize(ref_imgt.cpu()[0]).unsqueeze(0), revNormalize(frameT[0]).unsqueeze(0)],
                                       dim=0)
                else:
                    vImg = torch.cat([revNormalize(frame0[0]).unsqueeze(0), revNormalize(frame1[0]).unsqueeze(0),
                                        revNormalize(imgt.cpu()[0]).unsqueeze(0), revNormalize(frameT[0]).unsqueeze(0),
                                        revNormalize(ref_imgt.cpu()[0]).unsqueeze(0)],
                                       dim=0)

            vImg = torch.clamp(vImg, max=1., min=0)
            vis.images(vImg, win='vImage', env=args.visdom_env, nrow=2, opts={'title': 'visual_image'})


        # Validation and progress every `args.progress_iter` iterations
        if (trainIndex % args.progress_iter) == (args.progress_iter - 1) and not args.test:
            end = time.time()

            if args.final:
                detail_enhance_last.eval()
            else:
                if args.GEN_DE:
                    structure_gen.eval()
                else:
                    detail_enhance.eval()
            psnr, vLoss, valImg, MSE_val, ssim = validate()
            if args.final:
                detail_enhance_last.train()
            else:
                if args.GEN_DE:
                    structure_gen.train()
                else:
                    detail_enhance.train()

            valPSNR[epoch].append(psnr)
            valLoss[epoch].append(vLoss)

            itr = trainIndex + epoch * (len(trainloader))

            # Visdom

            vis.line(win='Loss', name='trainLoss', Y=torch.FloatTensor([iLoss / args.progress_iter]),
                     X=torch.FloatTensor([itr]), env=args.visdom_env, update='append')

            vis.line(win='Loss', name='validationLoss', Y=torch.FloatTensor([vLoss]),
                     X=torch.FloatTensor([itr]), env=args.visdom_env, update='append')

            vis.line(win='PSNR', name='PSNR', Y=torch.FloatTensor([psnr]), X=torch.FloatTensor([itr]),
                     env=args.visdom_env, update='append')

            vis.line(win='SSIM', name='SSIM', Y=torch.FloatTensor([ssim]), X=torch.FloatTensor([itr]),
                     env=args.visdom_env, update='append')

            vis.line(win='MSE_val', name='MSE_val', Y=torch.FloatTensor([MSE_val]), X=torch.FloatTensor([itr]),
                     env=args.visdom_env, update='append')

            # clamp for those value like: 1.0000012 (bug avoiding)
            valImg = torch.clamp(valImg, max=1.)
            vis.images(valImg, win='Image', env=args.visdom_env, nrow=2, opts={'title': 'img0_GT_SD_img1'})


            endVal = time.time()

            print(
                " Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  "
                "ValEvalTime: %0.2f LearningRate: %f" % (
                iLoss / args.progress_iter, trainIndex, len(trainloader), end - start, vLoss, psnr, endVal - end,
                get_lr(optimizer)))

            cLoss[epoch].append(iLoss / args.progress_iter)
            iLoss = 0
            start = time.time()


    # Create checkpoint after every `args.checkpoint_epoch` epochs
    if (epoch % args.checkpoint_epoch) == args.checkpoint_epoch - 1:
        dict1 = {
            'Detail': "End to end FeFlow.",
            'epoch': epoch,
            'timestamp': datetime.datetime.now(),
            'trainBatchSz': args.train_batch_size,
            'validationBatchSz': args.validation_batch_size,
            'learningRate': get_lr(optimizer),
            'loss': cLoss,
            'valLoss': valLoss,
            'valPSNR': valPSNR,
            'state_dictGEN': structure_gen.module.state_dict(),
            'state_dictDE': detail_enhance.module.state_dict()
            # 'state_dictDE_last': detail_enhance_last.module.state_dict()
        }
        torch.save(dict1, args.checkpoint_dir + "/FeFlow" + str(checkpoint_counter) + ".ckpt")
        checkpoint_counter += 1

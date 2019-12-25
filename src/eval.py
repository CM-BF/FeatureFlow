"""
Converts a Video to SeDraw version
"""
from time import time
import click
import cv2
import torch
from PIL import Image
import numpy as np
import src.model as model
import src.layers as layers
from torchvision import transforms
from torch.functional import F


torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_forward = transforms.ToTensor()
trans_backward = transforms.ToPILImage()
if device != "cpu":
    mean = [0.429, 0.431, 0.397]
    mea0 = [-m for m in mean]
    std = [1] * 3
    trans_forward = transforms.Compose([trans_forward, transforms.Normalize(mean=mean, std=std)])
    trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std), trans_backward])

network = layers.Network()
network = torch.nn.DataParallel(network)




def load_models(checkpoint):
    states = torch.load(checkpoint, map_location='cpu')
    network.module.load_state_dict(states['state_dictNET'])


def interpolate_batch(frames, factor):
    frame0 = torch.stack(frames[:-1])
    frame1 = torch.stack(frames[1:])

    img0 = frame0.to(device)
    img1 = frame1.to(device)

    frame_buffer = []
    for i in range(factor - 1):
        # start = time()
        frame_index = torch.ones(frames.__len__() - 1).type(torch.long) * i
        output = network((img0, img1, frame_index, None))
        ft_p = output[3]

        frame_buffer.append(ft_p)
        # print('time:', time() - start)

    return frame_buffer


def load_batch(video_in, batch_size, batch, w, h):
    if len(batch) > 0:
        batch = [batch[-1]]

    for i in range(batch_size):
        ok, frame = video_in.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize((w, h), Image.ANTIALIAS)
        frame = frame.convert('RGB')
        frame = trans_forward(frame)
        batch.append(frame)

    return batch


def denorm_frame(frame, w0, h0):
    frame = frame.cpu()
    frame = trans_backward(frame)
    frame = frame.resize((w0, h0), Image.BILINEAR)
    frame = frame.convert('RGB')
    return np.array(frame)[:, :, ::-1].copy()


def convert_video(source, dest, factor, batch_size=10, output_format='mp4v', output_fps=30):
    vin = cv2.VideoCapture(source)
    count = vin.get(cv2.CAP_PROP_FRAME_COUNT)
    w0, h0 = int(vin.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vin.get(cv2.CAP_PROP_FRAME_HEIGHT))

    codec = cv2.VideoWriter_fourcc(*output_format)
    vout = cv2.VideoWriter(dest, codec, float(output_fps), (w0, h0))

    w, h = (w0 // 32) * 32, (h0 // 32) * 32
    network.module.setup('custom', h, w)

    network.module.setup_t(factor)
    network.cuda()

    done = 0
    batch = []
    while True:
        batch = load_batch(vin, batch_size, batch, w, h)
        if len(batch) == 1:
            break
        done += len(batch) - 1

        intermediate_frames = interpolate_batch(batch, factor)
        intermediate_frames = list(zip(*intermediate_frames))

        for fid, iframe in enumerate(intermediate_frames):
            vout.write(denorm_frame(batch[fid], w0, h0))
            for frm in iframe:
                vout.write(denorm_frame(frm, w0, h0))

        try:
            yield len(batch), done, count
        except StopIteration:
            break

    vout.write(denorm_frame(batch[0], w0, h0))

    vin.release()
    vout.release()


@click.command('Evaluate Model by converting a low-FPS video to high-fps')
@click.argument('input')
@click.option('--checkpoint', help='Path to model checkpoint')
@click.option('--output', help='Path to output file to save')
@click.option('--batch', default=4, help='Number of frames to process in single forward pass')
@click.option('--scale', default=4, help='Scale Factor of FPS')
@click.option('--fps', default=30, help='FPS of output video')
def main(input, checkpoint, output, batch, scale, fps):
    avg = lambda x, n, x0: (x * n/(n+1) + x0 / (n+1), n+1)
    load_models(checkpoint)
    t0 = time()
    n0 = 0
    fpx = 0
    for dl, fd, fc in convert_video(input, output, int(scale), int(batch), output_fps=int(fps)):
        fpx, n0 = avg(fpx, n0, dl / (time() - t0))
        prg = int(100*fd/fc)
        eta = (fc - fd) / fpx
        print('\rDone: {:03d}% FPS: {:05.2f} ETA: {:.2f}s'.format(prg, fpx, eta) + ' '*5, end='')
        t0 = time()


if __name__ == '__main__':
    main()



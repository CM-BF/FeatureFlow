import os
import cv2
import numpy as np
import argparse
import shutil

import multiprocessing
from tqdm import tqdm

# command line parser
parser = argparse.ArgumentParser()

parser.add_argument('--videos_folder', type=str, required=True, help='the path to video dataset folder.')
parser.add_argument('--output_folder', type=str, default='../pre_dataset/', help='the path to output dataset folder.')
parser.add_argument('--lower_rate', type=int, default=5, help='lower the video fps by n times.')
args = parser.parse_args()


class DataCreator(object):

    def __init__(self):

        self.videos_folder = args.videos_folder
        self.output_folder = args.output_folder
        self.lower_rate = args.lower_rate
        self.tmp = '../.tmp/'
        try:
            os.mkdir(self.tmp)
        except:
            pass

    def _listener(self, pbar, q):
        for item in iter(q.get, None):
            pbar.update(1)

    def _lower_fps(self, p_args):
        video_name, q = p_args
        # pbar.set_description("Processing %s" % video_name)

        # read a video and create video_writer for lower fps video output
        video = cv2.VideoCapture(self.videos_folder + video_name)
        fps = video.get(cv2.CAP_PROP_FPS)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
        video_writer = [cv2.VideoWriter(self.tmp + video_name[:-4] + '_%s' % str(i) + '.mp4',
                                        fourcc,
                                        fps / self.lower_rate,
                                        size)
                        for i in range(self.lower_rate)]

        count = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                video_writer[count % self.lower_rate].write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
            else:
                break
            count += 1

        for i in range(self.lower_rate):
            video_writer[i].release()

        q.put(1)


    def lower_fps(self):

        videos_name = os.listdir(self.videos_folder)
        pbar = tqdm(total=len(videos_name))
        m = multiprocessing.Manager()
        q = m.Queue()

        listener = multiprocessing.Process(target=self._listener, args=(pbar, q))
        listener.start()

        p_args = [(video_name, q) for video_name in videos_name]
        pool = multiprocessing.Pool()
        pool.map(self._lower_fps, p_args)

        pool.close()
        pool.join()
        q.put(None)
        listener.join()

    def output(self):
        os.system('mkdir %s' % self.output_folder)
        os.system('cp %s %s' % (self.tmp + '*', self.output_folder))
        os.system('rm -rf %s' % self.tmp)


if __name__ == '__main__':
    data_creator = DataCreator()
    data_creator.lower_fps()
    data_creator.output()


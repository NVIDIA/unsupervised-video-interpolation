import sys
import os
import subprocess
import time
from inspect import isclass
import numpy as np


class TimerBlock:
    def __init__(self, title):
        print(("{}".format(title)))

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.clock()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")

    def log(self, string):
        duration = time.clock() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print("  [{:.3f}{}] {}".format(duration, units, string), flush=True)


def module_to_dict(module, exclude=[]):
    return dict([(x, getattr(module, x)) for x in dir(module)
                 if isclass(getattr(module, x))
                 and x not in exclude
                 and getattr(module, x) not in exclude])


# AverageMeter: adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# creat_pipe: adapted from https://stackoverflow.com/questions/23709893/popen-write-operation-on-closed-file-images-to-video-using-ffmpeg/23709937#23709937
# start an ffmpeg pipe for creating RGB8 for color images or FFV1 for depth
# NOTE: this is REALLY lossy and not optimal for HDR data. when it comes time to train
# on HDR data, you'll need to figure out the way to save to pix_fmt=rgb48 or something
# similar
def create_pipe(pipe_filename, width, height, frame_rate=60, quite=True):
    # default extension and tonemapper
    pix_fmt = 'rgb24'
    out_fmt = 'yuv420p'
    codec = 'h264'

    command = ['ffmpeg',
               '-threads', '2',  # number of threads to start
               '-y',  # (optional) overwrite output file if it exists
               '-f', 'rawvideo',  # input format
               '-vcodec', 'rawvideo',  # input codec
               '-s', str(width) + 'x' + str(height),  # size of one frame
               '-pix_fmt', pix_fmt,  # input pixel format
               '-r', str(frame_rate),  # frames per second
               '-i', '-',  # The imput comes from a pipe
               '-an',  # Tells FFMPEG not to expect any audio
               '-codec:v', codec,  # output codec
               '-crf', '18',
               # compression quality for h264 (maybe h265 too?) - http://slhck.info/video/2017/02/24/crf-guide.html
               # '-compression_level', '10', # compression level for libjpeg if doing lossy depth
               '-strict', '-2',  # experimental 16 bit support nessesary for gray16le
               '-pix_fmt', out_fmt,  # output pixel format
               '-s', str(width) + 'x' + str(height),  # output size
               pipe_filename]
    cmd = ' '.join(command)
    if not quite:
        print('openning a pip ....\n' + cmd + '\n')

    # open the pipe, and ignore stdout and stderr output
    DEVNULL = open(os.devnull, 'wb')
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=DEVNULL, stderr=DEVNULL, close_fds=True)



def get_pred_flag(height, width):
    pred_flag = np.ones((height, width, 3), dtype=np.uint8)
    pred_values = np.zeros((height, width, 3), dtype=np.uint8)

    hstart = int((192. / 1200) * height)
    wstart = int((224. / 1920) * width)
    h_step = int((24. / 1200) * height)
    w_step = int((32. / 1920) * width)

    pred_flag[hstart:hstart + h_step, -wstart + 0 * w_step:-wstart + 1 * w_step, :] = np.asarray([0, 0, 0])
    pred_flag[hstart:hstart + h_step, -wstart + 1 * w_step:-wstart + 2 * w_step, :] = np.asarray([0, 0, 0])
    pred_flag[hstart:hstart + h_step, -wstart + 2 * w_step:-wstart + 3 * w_step, :] = np.asarray([0, 0, 0])

    pred_values[hstart:hstart + h_step, -wstart + 0 * w_step:-wstart + 1 * w_step, :] = np.asarray([0, 0, 255])
    pred_values[hstart:hstart + h_step, -wstart + 1 * w_step:-wstart + 2 * w_step, :] = np.asarray([0, 255, 0])
    pred_values[hstart:hstart + h_step, -wstart + 2 * w_step:-wstart + 3 * w_step, :] = np.asarray([255, 0, 0])
    return pred_flag, pred_values


def copy_arguments(main_dict, main_filepath='', save_dir='./'):
    pycmd = 'python3 ' + main_filepath + ' \\\n'
    _main_dict = main_dict.copy()
    _main_dict['--name'] = _main_dict['--name']+'_replicate'
    for k in _main_dict.keys():
        if 'batchNorm' in k:
            pycmd += ' ' + k + ' ' + str(_main_dict[k]) + ' \\\n'
        elif type(_main_dict[k]) == bool and _main_dict[k]:
            pycmd += ' ' + k + ' \\\n'
        elif type(_main_dict[k]) == list:
            pycmd += ' ' + k + ' ' + ' '.join([str(f) for f in _main_dict[k]]) + ' \\\n'
        elif type(_main_dict[k]) != bool:
            pycmd += ' ' + k + ' ' + str(_main_dict[k]) + ' \\\n'
    pycmd = '#!/bin/bash\n' + pycmd[:-2]
    job_script = os.path.join(save_dir, 'job.sh')

    file = open(job_script, 'w')
    file.write(pycmd)
    file.close()

    return


def block_print():
    sys.stdout = open(os.devnull, 'w')

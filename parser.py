#!/usr/bin/env python3
# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import argparse
import models

# Collect all available model classes
model_names = sorted(el for el in models.__dict__
                     if not el.startswith("__") and callable(models.__dict__[el]))

"""
Reda, Fitsum A., et al. "Unsupervised Video Interpolation Using Cycle Consistency."
 arXiv preprint arXiv:1906.05928 (2019).

Jiang, Huaizu, et al. "Super slomo: High quality estimation of multiple
 intermediate frames for video interpolation." arXiv pre-print arXiv:1712.00080 (2017).
"""

parser = argparse.ArgumentParser(description="A PyTorch Implementation of Unsupervised Video Interpolation Using "
                                             "Cycle Consistency")

parser.add_argument('--model', metavar='MODEL', default='HJSuperSloMo',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: HJSuperSloMo)')
parser.add_argument('-s', '--save', '--save_root',
                    default='./result_folder', type=str,
                    help='Path of the output folder',
                    metavar='SAVE_PATH')
parser.add_argument('--torch_home', default='./.torch', type=str,
                    metavar='TORCH_HOME',
                    help='Path to save pre-trained models from torchvision')
parser.add_argument('-n', '--name', default='trial_0', type=str, metavar='EXPERIMENT_NAME',
                    help='Name of experiment folder.')
parser.add_argument('--dataset', default='VideoInterp', type=str, metavar='TRAINING_DATALOADER_CLASS',
                    help='Specify training dataset class for loading (Default: VideoInterp)')
parser.add_argument('--resume', default='', type=str, metavar='CHECKPOINT_PATH',
                    help='path to checkpoint file (default: none)')

# Resources
parser.add_argument('--distributed_backend', default='nccl', type=str, metavar='DISTRIBUTED_BACKEND',
                    help='backend used for communication between processes.')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loader workers (default: 10)')
parser.add_argument('-g', '--gpus', type=int, default=-1,
                    help='number of GPUs to use')
parser.add_argument('--fp16', action='store_true', help='Enable mixed-precision training.')

# Learning rate parameters.
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_scheduler', default='MultiStepLR', type=str,
                    metavar='LR_Scheduler', help='Scheduler for learning' +
                                                 ' rate (only ExponentialLR and MultiStepLR supported.')
parser.add_argument('--lr_gamma', default=0.1, type=float,
                    help='learning rate will be multiplied by this gamma')
parser.add_argument('--lr_step', default=200, type=int,
                    help='stepsize of changing the learning rate')
parser.add_argument('--lr_milestones', type=int, nargs='+',
                    default=[250, 450], help="Spatial dimension to " +
                                             "crop training samples for training")
# Gradient.
parser.add_argument('--clip_gradients', default=-1.0, type=float,
                    help='If positive, clip the gradients by this value.')

# Optimization hyper-parameters
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='BATCH_SIZE',
                    help='mini-batch per gpu size (default : 4)')
parser.add_argument('--wd', '--weight_decay', default=0.001, type=float, metavar='WEIGHT_DECAY',
                    help='weight_decay (default = 0.001)')
parser.add_argument('--seed', default=1234, type=int, metavar="SEED",
                    help='seed for initializing training. ')
parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPTIMIZER',
                    help='Specify optimizer from torch.optim (Default: Adam)')
parser.add_argument('--mean_pix', nargs='+', type=float, metavar="RGB_MEAN",
                    default=[109.93, 109.167, 101.455],
                    help='mean pixel values carried over from superslomo (default: [109.93, 109.167, 101.455])')
parser.add_argument('--print_freq', default=100, type=int, metavar="PRINT_FREQ",
                    help='frequency of printing training status (default: 100)')
parser.add_argument('--save_freq', type=int, default=20, metavar="SAVE_FREQ",
                    help='frequency of saving intermediate models, in epoches (default: 20)')
parser.add_argument('--start_epoch', type=int, default=-1,
                    help="Set epoch number during resuming")
parser.add_argument('--epochs', default=500, type=int, metavar="EPOCHES",
                    help='number of total epochs to run (default: 500)')

# Training sequence, supports a single sequence for now
parser.add_argument('--train_file', required=False, metavar="TRAINING_FILE",
                    help='training file (default : Required)')
parser.add_argument('--crop_size', type=int, nargs='+', default=[704, 704], metavar="CROP_SIZE",
                    help="Spatial dimension to crop training samples for training (default : [704, 704])")
parser.add_argument('--train_n_batches', default=-1, type=int, metavar="TRAIN_N_BATCHES",
                    help="Limit the number of minibatch iterations per epoch. Used for debugging purposes. \
                    (default : -1, means use all available mini-batches")
parser.add_argument('--sample_rate', type=int, default=1,
                    help='number of frames to skip when sampling input1, {intermediate}, and input2 \
                    (default=1, ie. we treat consecutive frames for input1 and intermediate, and input2 frames.)')
parser.add_argument('--step_size', type=int, default=-1, metavar="STEP_INTERP",
                    help='number of frames to skip from one mini-batch to the next mini-batch \
                    (default -1, means step_size = num_interp + 1')
parser.add_argument('--num_interp', default=7, type=int, metavar="NUM_INTERP",
                    help='number intermediate frames to interpolate (default : 7)')


# Validation sequence, supports a single sequence for now
parser.add_argument('--val_file', metavar="VALIDATION_FILE",
                    help='validation file (default : None)')
parser.add_argument('--val_batch_size', type=int, default=1,
                    help="Batch size to use for validation.")
parser.add_argument('--val_n_batches', default=-1, type=int,
                    help="Limit the number of minibatch iterations per epoch. Used for debugging purposes.")
parser.add_argument('--video_fps', type=int, default=30,
                    help="Render predicted video with a specified frame rate")
parser.add_argument('--initial_eval', action='store_true', help='Perform initial evaluation before training.')
parser.add_argument("--start_index", type=int, default=0, metavar="VAL_START_INDEX",
                    help="Index to start running validation (default : 0)")
parser.add_argument("--val_sample_rate", type=int, default=1, metavar="VAL_START_INDEX",
                    help='number of frames to skip when sampling input1, {intermediate}, and input2 (default=1, \
                     ie. we treat consecutive frames for input1 and intermediate, and input2 frames.)')
parser.add_argument('--val_step_size', type=int, default=-1, metavar="VAL_STEP_INTERP",
                    help='number of frames to skip from one mini-batch to the next mini-batch \
                    (default -1, means step_size = num_interp + 1')
parser.add_argument('--val_num_interp', type=int, default=1,
                    help='number of intermediate frames we want to interpolate for validation. (default: 1)')

# Misc: undersample large sequences (--step_size), compute flow after downscale (--flow_scale)
parser.add_argument('--flow_scale', type=float, default=1.,
                    help="Flow scale (default: 1.) for robust interpolation in high resolution images.")
parser.add_argument('--skip_aug', action='store_true', help='Skips expensive geometric or photometric augmentations.')
parser.add_argument('--teacher_weight', type=float, default=-1.,
                    help="Teacher or Pseudo Supervised Loss (PSL)'s weight of contribution to total loss.")

parser.add_argument('--apply_vidflag', action='store_true', help='Apply applying the BRG flag to interpolated frames.')

parser.add_argument('--write_video', action='store_true', help='save video to \'args.save/args.name.mp4\'.')
parser.add_argument('--write_images', action='store_true',
                    help='write to folder \'args.save/args.name\' prediction and ground-truth images.')
parser.add_argument('--stride', type=int, default=64,
                    help='the largest factor a model reduces spatial size of inputs during a forward pass.')
parser.add_argument('--post_fix', default='Proposed', type=str,
                    help='tag for predicted frames (default: \'proposed\')')

# Required for torch distributed launch
parser.add_argument('--local_rank', default=None, type=int,
                    help='Torch Distributed')

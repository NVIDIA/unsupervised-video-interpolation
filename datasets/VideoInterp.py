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
from __future__ import division
from __future__ import print_function

import os
import natsort
import numpy as np
from imageio import imread
import torch
from torch.utils import data


class VideoInterp(data.Dataset):
    def __init__(self, args=None, root='', num_interp=7, sample_rate=1, step_size=1,
                 is_training=False, transform=None):

        self.num_interp = num_interp
        self.sample_rate = sample_rate
        self.step_size = step_size
        self.transform = transform
        self.is_training = is_training
        self.transform = transform

        self.start_index = args.start_index
        self.stride = args.stride
        self.crop_size = args.crop_size

        # argument sanity check
        assert (os.path.exists(root)), "Invalid path to input dataset."
        assert self.num_interp > 0, "num_interp must be at least 1"
        assert self.step_size > 0, "step_size must be at least 1"

        if self.is_training:
            self.start_index = 0

        # collect, colors, motion vectors, and depth
        self.ref = self.collect_filelist(root)

        # calculate total number of unique sub-sequences
        def calc_subseq_len(n):
            return (n - max(1, (self.num_interp + 1) * self.sample_rate) - 1) // self.step_size + 1
        self.counts = [calc_subseq_len(len(el)) for el in self.ref]

        self.total = np.sum(self.counts)
        self.cum_sum = list(np.cumsum([0] + [el for el in self.counts]))

    def collect_filelist(self, root):
        include_ext = [".png", ".jpg", "jpeg", ".bmp"]
        # collect subfolders, excluding hidden files, but following symlinks
        dirs = [x[0] for x in os.walk(root, followlinks=True) if not x[0].startswith('.')]

        # naturally sort, both dirs and individual images, while skipping hidden files
        dirs = natsort.natsorted(dirs)

        datasets = [
            [os.path.join(fdir, el) for el in natsort.natsorted(os.listdir(fdir))
             if os.path.isfile(os.path.join(fdir, el))
             and not el.startswith('.')
             and any([el.endswith(ext) for ext in include_ext])]
            for fdir in dirs
        ]

        return [el for el in datasets if el]

    def get_sample_indices(self, index, tar_index=None):
        if self.is_training:
            sample_indices = [index, index + self.sample_rate * tar_index, index +
                              self.sample_rate * (self.num_interp + 1)]
        else:
            sample_indices = [index + i * self.sample_rate for i in range(0, self.num_interp + 2)]
            if self.sample_rate == 0:
                sample_indices[-1] += 1
        return sample_indices

    def pad_images(self, images):
        height, width, _ = images[0].shape
        image_count = len(images)
        # Pad images with zeros if it is not evenly divisible by args.stride (property of model)
        if (height % self.stride) != 0:
            new_height = (height // self.stride + 1) * self.stride
            for i in range(image_count):
                images[i] = np.pad(images[i], ((0, new_height - height), (0, 0), (0, 0)), 'constant',
                                   constant_values=(0, 0))

        if (width % self.stride) != 0:
            new_width = (width // self.stride + 1) * self.stride
            for i in range(image_count):
                images[i] = np.pad(images[i], ((0, 0), (0, new_width - width), (0, 0)), 'constant',
                                   constant_values=(0, 0))
        return images

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        # Adjust index
        index = len(self) + index if index < 0 else index
        index = index + self.start_index

        dataset_index = np.searchsorted(self.cum_sum, index + 1)
        index = self.step_size * (index - self.cum_sum[np.maximum(0, dataset_index - 1)])

        image_list = self.ref[dataset_index - 1]

        # target index, subset of range(1,num_interp+1)
        tar_index = 1 + torch.randint(0, max(1, self.num_interp), (1,)).item()
        input_indices = self.get_sample_indices(index, tar_index)

        # reverse subsequence for augmentation with a probability of 0.5
        if self.is_training and torch.randint(0, 2, (1,)).item():
            input_indices = input_indices[::-1]
            tar_index = self.num_interp - tar_index + 1

        image_files = [image_list[i] for i in input_indices]

        # Read images from file
        images = [imread(image_file)[:, :, :3] for image_file in image_files]
        image_shape = images[0].shape

        # Apply data augmentation if defined.
        if self.transform:
            input_images, target_images = [images[0], images[-1]], images[1:-1]
            input_images, target_images = self.transform(input_images, target_images)
            images = [input_images[0]] + target_images + [input_images[-1]]

        # Pad images with zeros, so they fit evenly to model arch in forward pass.
        padded_images = self.pad_images(images)

        input_images = [torch.from_numpy(np.ascontiguousarray(tmp.transpose(2, 0, 1).astype(np.float32))).float() for
                        tmp in padded_images]

        output_dict = {
            'image': input_images, 'tindex': tar_index, 'ishape': image_shape[:2], 'input_files': image_files
        }
        # print (' '.join([os.path.basename(f) for f in image_files]))
        return output_dict


class CycleVideoInterp(VideoInterp):
    def __init__(self, args=None, root='', num_interp=7, sample_rate=1, step_size=1,
                 is_training=False, transform=None):
        super(CycleVideoInterp, self).__init__(args=args, root=root, num_interp=num_interp, sample_rate=sample_rate,
                                               step_size=step_size, is_training=is_training, transform=transform)

        # # Adjust  indices
        if self.is_training:
            self.counts = [el - 1 for el in self.counts]
        self.total = np.sum(self.counts)
        self.cum_sum = list(np.cumsum([0] + [el for el in self.counts]))

    def get_sample_indices(self, index, tar_index=None):
        if self.is_training:
            offset = max(1, self.sample_rate) + self.sample_rate * self.num_interp
            sample_indices = [index, index + offset, index + 2 * offset]
        else:
            sample_indices = [index + i * self.sample_rate for i in range(0, self.num_interp + 2)]
            if self.sample_rate == 0:
                sample_indices[-1] += 1
        return sample_indices

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The baseline Super SloMo relies on torch.nn.functional.grid_sample to implement a warping module.
# To ensure that our results replicate published accuracy numbers, we also implement a Resample2D layer
# in a similar way, completely with torch tensors, as is done in:
# https://github.com/avinashpaliwal/Super-SloMo/blob/master/model.py#L213
#
# However, for faster training, we suggest to use our CUDA kernels for Resample2D, here:
# https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/resample2d_package/resample2d.py
#
# from flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d
#


class MyResample2D(nn.Module):
    def __init__(self, width, height):
        super(MyResample2D, self).__init__()

        self.width = width
        self.height = height

        # make grids for horizontal and vertical displacements
        grid_w, grid_h = np.meshgrid(np.arange(width), np.arange(height))
        grid_w, grid_h = grid_w.reshape((1,) + grid_w.shape), grid_h.reshape((1,) + grid_h.shape)

        self.register_buffer("grid_w", torch.tensor(grid_w, requires_grad=False, dtype=torch.float32))
        self.register_buffer("grid_h", torch.tensor(grid_h, requires_grad=False, dtype=torch.float32))

    def forward(self, im, uv):

        # Get relative displacement
        u = uv[:, 0, ...]
        v = uv[:, 1, ...]

        # Calculate absolute displacement along height and width axis -> (batch_size, height, width)
        ww = self.grid_w.expand_as(u) + u
        hh = self.grid_h.expand_as(v) + v

        # Normalize indices to [-1,1]
        ww = 2 * ww / (self.width - 1) - 1
        hh = 2 * hh / (self.height - 1) - 1

        # Form a grid of shape (batch_size, height, width, 2)
        norm_grid_wh = torch.stack((ww, hh), dim=-1)

        # Perform a resample
        reampled_im = torch.nn.functional.grid_sample(im, norm_grid_wh)

        return reampled_im


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

    def forward(self, inputs, target_index):
        return {}, inputs['image'][1], inputs['image'][1]

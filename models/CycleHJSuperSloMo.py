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
import torch
import torch.nn.functional as F
from .model_utils import MyResample2D, DummyModel
from .HJSuperSloMo import HJSuperSloMo


class CycleHJSuperSloMo(HJSuperSloMo):
    def __init__(self, args, mean_pix=[109.93, 109.167, 101.455]):
        super(CycleHJSuperSloMo, self).__init__(args=args, mean_pix=mean_pix)

        if args.resume:
            self.teacher = HJSuperSloMo(args)
            checkpoint = torch.load(args.resume, map_location='cpu')
            self.teacher.load_state_dict(checkpoint['state_dict'], strict=False)
            for param in self.teacher.parameters():
                param.requires_grad = False

            self.teacher_weight = 0.8
            if 'teacher_weight' in args and args.teacher_weight >= 0:
                self.teacher_weight = args.teacher_weight
        else:
            self.teacher = DummyModel()
            self.teacher_weight = 0.

    def network_output(self, inputs, target_index):

        im1, im2 = inputs

        # Estimate bi-directional optical flows between input low FPS frame pairs
        # Downsample images for robust intermediate flow estimation
        ds_im1 = F.interpolate(im1, scale_factor=1./self.scale, mode='bilinear', align_corners=False)
        ds_im2 = F.interpolate(im2, scale_factor=1./self.scale, mode='bilinear', align_corners=False)

        uvf, bottleneck_out, uvb = self.make_flow_prediction(torch.cat((ds_im1, ds_im2), dim=1))

        uvf = self.scale * F.interpolate(uvf, scale_factor=self.scale, mode='bilinear', align_corners=False)
        uvb = self.scale * F.interpolate(uvb, scale_factor=self.scale, mode='bilinear', align_corners=False)
        bottleneck_out = F.interpolate(bottleneck_out, scale_factor=self.scale, mode='bilinear', align_corners=False)

        t = self.tlinespace[target_index]
        t = t.reshape(t.shape[0], 1, 1, 1)

        uvb_t_raw = - (1 - t) * t * uvf + t * t * uvb
        uvf_t_raw = (1 - t) * (1 - t) * uvf - (1 - t) * t * uvb

        im1w_raw = self.resample2d(im1, uvb_t_raw)  # im1w_raw
        im2w_raw = self.resample2d(im2, uvf_t_raw)  # im2w_raw

        # Perform intermediate bi-directional flow refinement
        uv_t_data = torch.cat((im1, im2, im1w_raw, uvb_t_raw, im2w_raw, uvf_t_raw), dim=1)
        uvf_t, uvb_t, t_vis_map = self.make_flow_interpolation(uv_t_data, bottleneck_out)

        uvb_t = uvb_t_raw + uvb_t # uvb_t
        uvf_t = uvf_t_raw + uvf_t # uvf_t

        im1w = self.resample2d(im1, uvb_t)  # im1w
        im2w = self.resample2d(im2, uvf_t)  # im2w

        # Compute final intermediate frame via weighted blending
        alpha1 = (1 - t) * t_vis_map
        alpha2 = t * (1 - t_vis_map)
        denorm = alpha1 + alpha2 + 1e-10
        im_t_out = (alpha1 * im1w + alpha2 * im2w) / denorm

        return im_t_out, uvb, uvf

    def network_eval(self, inputs, target_index):
        _, _, height, width = inputs[0].shape
        self.resample2d = MyResample2D(width, height).cuda()

        # Normalize inputs
        im1, im_target, im2 = [(im - self.mean_pix) for im in inputs]

        im_t_out, uvb, uvf = self.network_output([im1, im2], target_index)

        # Calculate losses
        losses = {}
        losses['pix_loss'] = self.L1_loss(im_t_out, im_target)

        im_t_out_features = self.vgg16_features(im_t_out / 255.)
        im_target_features = self.vgg16_features(im_target / 255.)
        losses['vgg16_loss'] = self.L2_loss(im_t_out_features, im_target_features)

        losses['warp_loss'] = self.L1_loss(self.resample2d(im1, uvb.contiguous()), im2) + \
            self.L1_loss(self.resample2d(im2, uvf.contiguous()), im1)

        smooth_bwd = self.L1_loss(uvb[:, :, :, :-1], uvb[:, :, :, 1:]) + \
            self.L1_loss(uvb[:, :, :-1, :], uvb[:, :, 1:, :])
        smooth_fwd = self.L1_loss(uvf[:, :, :, :-1], uvf[:, :, :, 1:]) + \
            self.L1_loss(uvf[:, :, :-1, :], uvf[:, :, 1:, :])

        losses['smooth_loss'] = smooth_bwd + smooth_fwd

        # Coefficients for total loss determined empirically using a validation set
        losses['tot'] = 0.8 * losses['pix_loss'] + 0.4 * losses['warp_loss'] + 0.005 * losses['vgg16_loss'] + losses[
            'smooth_loss']

        # Converts back to (0, 255) range
        im_t_out = im_t_out + self.mean_pix
        im_target = im_target + self.mean_pix

        return losses, im_t_out, im_target

    def forward(self, inputs, target_index):
        if 'image' in inputs:
            inputs = inputs['image']

        if not self.training:
            return self.network_eval(inputs, target_index)
        self.resample2d = MyResample2D(inputs[0].shape[-1], inputs[0].shape[-2]).cuda()

        # Input frames
        im1, im2, im3 = inputs

        # Calculate Pseudo targets at interm_index
        with torch.no_grad():
            _, psuedo_gt12, _ = self.teacher({'image': [im1, im1, im2]}, target_index)
            _, psuedo_gt23, _ = self.teacher({'image': [im2, im3, im3]}, target_index)
        psuedo_gt12, psuedo_gt23 = psuedo_gt12 - self.mean_pix, psuedo_gt23 - self.mean_pix

        im1, im2, im3 = im1 - self.mean_pix, im2 - self.mean_pix, im3 - self.mean_pix

        pred12, pred12_uvb, pred12_uvf = self.network_output([im1, im2], target_index)
        pred23, pred23_uvb, pred23_uvf = self.network_output([im2, im3], target_index)

        target_index = (self.args.num_interp + 1) - target_index

        ds_pred12 = F.interpolate(pred12, scale_factor=1./self.scale, mode='bilinear', align_corners=False)
        ds_pred23 = F.interpolate(pred23, scale_factor=1./self.scale, mode='bilinear', align_corners=False)

        uvf, bottleneck_out, uvb = self.make_flow_prediction(torch.cat((ds_pred12, ds_pred23), dim=1))

        uvf = self.scale * F.interpolate(uvf, scale_factor=self.scale, mode='bilinear', align_corners=False)
        uvb = self.scale * F.interpolate(uvb, scale_factor=self.scale, mode='bilinear', align_corners=False)
        bottleneck_out = F.interpolate(bottleneck_out, scale_factor=self.scale, mode='bilinear', align_corners=False)

        t = self.tlinespace[target_index]
        t = t.reshape(t.shape[0], 1, 1, 1)

        uvb_t_raw = - (1 - t) * t * uvf + t * t * uvb
        uvf_t_raw = (1 - t) * (1 - t) * uvf - (1 - t) * t * uvb

        im12w_raw = self.resample2d(pred12, uvb_t_raw)  # im1w_raw
        im23w_raw = self.resample2d(pred23, uvf_t_raw)  # im2w_raw

        # Perform intermediate bi-directional flow refinement
        uv_t_data = torch.cat((pred12, pred23, im12w_raw, uvb_t_raw, im23w_raw, uvf_t_raw), dim=1)
        uvf_t, uvb_t, t_vis_map = self.make_flow_interpolation(uv_t_data, bottleneck_out)

        uvb_t = uvb_t_raw + uvb_t # uvb_t
        uvf_t = uvf_t_raw + uvf_t # uvf_t

        im12w = self.resample2d(pred12, uvb_t)  # im1w
        im23w = self.resample2d(pred23, uvf_t)  # im2w

        # Compute final intermediate frame via weighted blending
        alpha1 = (1 - t) * t_vis_map
        alpha2 = t * (1 - t_vis_map)
        denorm = alpha1 + alpha2 + 1e-10
        im_t_out = (alpha1 * im12w + alpha2 * im23w) / denorm

        # Calculate training loss
        losses = {}
        losses['pix_loss'] = self.L1_loss(im_t_out, im2)

        im_t_out_features = self.vgg16_features(im_t_out/255.)
        im2_features = self.vgg16_features(im2/255.)
        losses['vgg16_loss'] = self.L2_loss(im_t_out_features, im2_features)

        losses['warp_loss'] = self.L1_loss(im12w_raw, im2) + self.L1_loss(im23w_raw, im2) + \
            self.L1_loss(self.resample2d(pred12, uvb), pred23) + \
            self.L1_loss(self.resample2d(pred23, uvf), pred12) + \
            self.L1_loss(self.resample2d(im1, pred12_uvb), im2) + \
            self.L1_loss(self.resample2d(im2, pred12_uvf), im1) + \
            self.L1_loss(self.resample2d(im2, pred23_uvb), im3) + \
            self.L1_loss(self.resample2d(im3, pred23_uvf), im2)

        smooth_bwd = self.L1_loss(uvb[:, :, :, :-1], uvb[:, :, :, 1:]) + \
            self.L1_loss(uvb[:, :, :-1, :], uvb[:, :, 1:, :]) + \
            self.L1_loss(pred12_uvb[:, :, :, :-1], pred12_uvb[:, :, :, 1:]) + \
            self.L1_loss(pred12_uvb[:, :, :-1, :], pred12_uvb[:, :, 1:, :]) + \
            self.L1_loss(pred23_uvb[:, :, :, :-1], pred23_uvb[:, :, :, 1:]) + \
            self.L1_loss(pred23_uvb[:, :, :-1, :], pred23_uvb[:, :, 1:, :])

        smooth_fwd = self.L1_loss(uvf[:, :, :, :-1], uvf[:, :, :, 1:]) + \
            self.L1_loss(uvf[:, :, :-1, :], uvf[:, :, 1:, :]) + \
            self.L1_loss(pred12_uvf[:, :, :, :-1], pred12_uvf[:, :, :, 1:]) + \
            self.L1_loss(pred12_uvf[:, :, :-1, :], pred12_uvf[:, :, 1:, :]) + \
            self.L1_loss(pred23_uvf[:, :, :, :-1], pred23_uvf[:, :, :, 1:]) + \
            self.L1_loss(pred23_uvf[:, :, :-1, :], pred23_uvf[:, :, 1:, :])

        losses['loss_smooth'] = smooth_bwd + smooth_fwd

        losses['teacher'] = self.L1_loss(psuedo_gt12, pred12) + self.L1_loss(psuedo_gt23, pred23)

        # Coefficients for total loss determined empirically using a validation set
        losses['tot'] = self.pix_alpha * losses['pix_loss'] + self.warp_alpha * losses['warp_loss'] + \
            self.vgg16_alpha * losses['vgg16_loss'] + self.smooth_alpha * losses['loss_smooth'] + self.teacher_weight * losses['teacher']

        # Converts back to (0, 255) range
        im_t_out = im_t_out + self.mean_pix
        im_target = im2 + self.mean_pix

        return losses, im_t_out, im_target

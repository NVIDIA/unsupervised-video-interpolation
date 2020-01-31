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
import os
import sys
import shutil
import natsort
import numpy as np
from glob import glob
from imageio import imsave
from skimage.measure import compare_psnr, compare_ssim
from tqdm import tqdm
tqdm.monitor_interval = 0

import torch
import torch.backends.cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from parser import parser
import datasets
import models
import utils

"""
Reda, Fitsum A., et al. "Unsupervised Video Interpolation Using Cycle Consistency."
 arXiv preprint arXiv:1906.05928 (2019).
  
Jiang, Huaizu, et al. "Super slomo: High quality estimation of multiple
 intermediate frames for video interpolation." arXiv pre-print arXiv:1712.00080 (2017).
"""


def main():
    with utils.TimerBlock("\nParsing Arguments") as block:
        args = parser.parse_args()

        args.rank = int(os.getenv('RANK', 0))

        block.log("Creating save directory: {}".format(args.save))
        args.save_root = os.path.join(args.save, args.name)
        if args.write_images or args.write_video:
            os.makedirs(args.save_root, exist_ok=True)
            assert os.path.exists(args.save_root)
        else:
            os.makedirs(args.save, exist_ok=True)
            assert os.path.exists(args.save)

        os.makedirs(args.torch_home, exist_ok=True)
        os.environ['TORCH_HOME'] = args.torch_home

        args.gpus = torch.cuda.device_count() if args.gpus < 0 else args.gpus
        block.log('Number of gpus: {} | {}'.format(args.gpus, list(range(args.gpus))))

        args.network_class = utils.module_to_dict(models)[args.model]
        args.dataset_class = utils.module_to_dict(datasets)[args.dataset]
        block.log('save_root: {}'.format(args.save_root))
        block.log('val_file: {}'.format(args.val_file))

    with utils.TimerBlock("Building {} Dataset".format(args.dataset)) as block:
        vkwargs = {'batch_size': args.gpus * args.val_batch_size,
                   'num_workers': args.gpus * args.workers,
                   'pin_memory': True, 'drop_last': True}
        step_size = args.val_step_size if args.val_step_size > 0 else (args.num_interp + 1)
        val_dataset = args.dataset_class(args=args, root=args.val_file, num_interp=args.num_interp,
                                         sample_rate=args.val_sample_rate, step_size=step_size)

        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
                                                 **vkwargs)

        args.folder_list = natsort.natsorted(
            [os.path.basename(f) for f in sorted(glob(os.path.join(args.val_file, '*')))])

        block.log('Number of Validation Images: {}:({} mini-batches)'.format(len(val_loader.dataset), len(val_loader)))

    with utils.TimerBlock("Building {} Model".format(args.model)) as block:
        model = args.network_class(args)

        block.log('Number of parameters: {val:,}'.format(val=
            sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])))

        block.log('Initializing CUDA')
        assert torch.cuda.is_available(), 'Code supported for GPUs only at the moment'
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus)))
        torch.manual_seed(args.seed)

        block.log("Attempting to Load checkpoint '{}'".format(args.resume))
        if args.resume and os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)

            # Partial initialization
            input_dict = checkpoint['state_dict']
            curr_dict = model.module.state_dict()
            state_dict = input_dict.copy()
            for key in input_dict:
                if key not in curr_dict:
                    continue
                if curr_dict[key].shape != input_dict[key].shape:
                    state_dict.pop(key)
                    print("key {} skipped because of size mismatch.".format(key))
            model.module.load_state_dict(state_dict, strict=False)

            epoch = checkpoint['epoch']
            block.log("Successfully loaded checkpoint (at epoch {})".format(epoch))
        elif args.resume:
            block.log("No checkpoint found at '{}'.\nAborted.".format(args.resume))
            sys.exit(0)
        else:
            block.log("Random initialization, checkpoint not provided.")

    with utils.TimerBlock("Inference started ") as block:
        evaluate(args, val_loader, model, args.num_interp, epoch, block)


def evaluate(args, val_loader, model, num_interp, epoch, block):
    in_height, in_width = val_loader.dataset[0]['ishape']
    pred_flag, pred_values = utils.get_pred_flag(in_height, in_width)

    if not args.apply_vidflag:
        pred_flag = 0 * pred_flag + 1
        pred_values = 0 * pred_values

    if args.rank == 0 and args.write_video:
        video_file = os.path.join(args.save_root, '__epoch_%03d.mp4' % epoch)
        _pipe = utils.create_pipe(video_file, in_width, in_height, frame_rate=args.video_fps)

    model.eval()

    loss_values = utils.AverageMeter()
    avg_metrics = np.zeros((0, 3), dtype=float)
    num_batches = len(val_loader) if args.val_n_batches < 0 else args.val_n_batches

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=num_batches)):

            inputs = [b.cuda() for b in batch['image']]

            input_images = [inputs[0], inputs[len(inputs) // 2], inputs[-1]]
            inputs_dict = {'image': input_images}

            target_images = inputs[1:-1]
            tar_indices = batch['tindex'].cuda()

            # compute loss at mid-way
            tar_indices[:] = (num_interp + 1) // 2
            loss, outputs, _ = model(inputs_dict, tar_indices)
            loss_values.update(loss['tot'].data.item(), outputs.size(0))

            # compute output for each intermediate timepoint
            output_image = inputs[0]
            for tarIndex in range(1, num_interp + 1):
                tar_indices[:] = tarIndex
                _, outputs, _ = model(inputs_dict, tar_indices)
                output_image = torch.cat((output_image, outputs), dim=1)
            output_image = torch.split(output_image, 3, dim=1)[1:]

            batch_size, _, _, _ = inputs[0].shape
            input_filenames = batch['input_files'][1:-1]
            in_height, in_width = batch['ishape']

            for b in range(batch_size):
                first_target = (input_images[0][b].data.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
                first_target = first_target[:in_height, :in_width, :]
                second_target = (input_images[-1][b].data.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
                second_target = second_target[:in_height, :in_width, :]

                gt_image = first_target
                for index in range(num_interp):
                    pred_image = (output_image[index][b].data.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
                    pred_image = pred_image[:in_height, :in_width, :]

                    # if ground-truth not loaded, treat low FPS frames as targets
                    if index < len(target_images):
                        gt_image = (target_images[index][b].data.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
                        gt_filename = '/'.join(input_filenames[index][b].split(os.sep)[-2:])
                    gt_image = gt_image[:in_height, :in_width, :]

                    # calculate metrics using skimage
                    psnr = compare_psnr(pred_image, gt_image)
                    ssim = compare_ssim(pred_image, gt_image, multichannel=True, gaussian_weights=True)
                    err = pred_image.astype(np.float32) - gt_image.astype(np.float32)
                    ie = np.mean(np.sqrt(np.sum(err * err, axis=2)))

                    avg_metrics = np.vstack((avg_metrics, np.array([psnr, ssim, ie])))

                    # write_images
                    if args.write_images:
                        tmp_filename = os.path.join(args.save_root, "%s-%02d-%s.png" % (gt_filename[:-4], (index + 1), args.post_fix))
                        os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)
                        imsave(tmp_filename, pred_image)

                    # write video
                    if args.rank == 0 and args.write_video:
                        if index == 0:
                            _pipe.stdin.write(first_target.tobytes())
                        try:
                            _pipe.stdin.write((pred_image * pred_flag + pred_values).tobytes())
                        except AttributeError:
                            raise AttributeError("Error in ffmpeg video creation. Inconsistent image size.")
                if args.write_images:
                    tmp_filename = os.path.join(args.save_root, "%s-%02d-%s.png" % (gt_filename[:-4], 0, "ground_truth"))
                    os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)
                    imsave(tmp_filename, first_target)
                    tmp_filename = os.path.join(args.save_root, "%s-%02d-%s.png" % (gt_filename[:-4], num_interp+1, "ground_truth"))
                    imsave(tmp_filename, second_target)
            if (i + 1) >= num_batches:
                break

    if args.write_video:
        _pipe.stdin.close()
        _pipe.wait()

    """
    Print final accuracy statistics. If intermediate ground truth frames are not available from the input sequence, 
    the first low FPS frame is treated as a ground-truth frame for all intermediately predicted frames, 
    as the quantities should not be trusted, in this case.
    """
    for i in range(num_interp):
        result2print = 'interm {:02d} PSNR: {:.2f}, SSIM: {:.3f}, IE: {:.2f}'.format(i+1,
            np.nanmean(avg_metrics[i::num_interp], axis=0)[0],
            np.nanmean(avg_metrics[i::num_interp], axis=0)[1],
            np.nanmean(avg_metrics[i::num_interp], axis=0)[2])
        block.log(result2print)

    avg_metrics = np.nanmean(avg_metrics, axis=0)
    result2print = 'Overall PSNR: {:.2f}, SSIM: {:.3f}, IE: {:.2f}'.format(avg_metrics[0], avg_metrics[1],
                                                                           avg_metrics[2])
    v_psnr, v_ssim, v_ie = avg_metrics[0], avg_metrics[1], avg_metrics[2]
    block.log(result2print)

    # re-name video with psnr
    if args.rank == 0 and args.write_video:
        shutil.move(os.path.join(args.save_root, '__epoch_%03d.mp4' % epoch),
                    os.path.join(args.save_root, '__epoch_%03d_psnr_%1.2f.mp4' % (epoch, avg_metrics[0])))

    # Move back the model to train mode.
    model.train()

    torch.cuda.empty_cache()
    block.log('max memory allocated (GB): {:.3f}: '.format(
        torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))

    return v_psnr, v_ssim, v_ie, loss_values.val


if __name__ == '__main__':
    main()

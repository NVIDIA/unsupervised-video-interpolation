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
import random
import math
import numpy as np

import torch.backends.cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm
tqdm.monitor_interval = 0

import datasets
import models
import utils
from parser import parser
from eval import evaluate
from datasets import data_transforms

# Import apex's distributed module. 
try:
    from apex.parallel import DistributedDataParallel
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from apex import amp

"""
Reda, Fitsum A., et al. "Unsupervised Video Interpolation Using Cycle Consistency."
 arXiv preprint arXiv:1906.05928 (2019).

Jiang, Huaizu, et al. "Super slomo: High quality estimation of multiple
 intermediate frames for video interpolation." arXiv pre-print arXiv:1712.00080 (2017).
"""


def parse_and_set_args(block):
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    block.log("Enabling torch.backends.cudnn.benchmark")

    if args.resume != '':
        block.log("Setting initial eval to true since checkpoint is provided")
        args.initial_eval = True

    args.rank = int(os.getenv('RANK', 0))
    args.world_size = int(os.getenv("WORLD_SIZE", 1))

    if args.local_rank:
        args.rank = args.local_rank
    if args.local_rank is not None and args.local_rank != 0:
        utils.block_print()

    block.log("Creating save directory: {}".format(
        os.path.join(args.save, args.name)))
    args.save_root = os.path.join(args.save, args.name)
    os.makedirs(args.save_root, exist_ok=True)
    assert os.path.exists(args.save_root)

    # temporary directory for torch pre-trained models
    os.makedirs(args.torch_home, exist_ok=True)
    os.environ['TORCH_HOME'] = args.torch_home

    defaults, input_arguments = {}, {}
    for key in vars(args):
        defaults[key] = parser.get_default(key)

    for argument, value in sorted(vars(args).items()):
        if value != defaults[argument] and argument in vars(parser.parse_args()).keys():
            input_arguments['--' + str(argument)] = value
            block.log('{}: {}'.format(argument, value))

    if args.rank == 0:
        utils.copy_arguments(input_arguments, os.path.realpath(__file__),
                             args.save_root)

    args.network_class = utils.module_to_dict(models)[args.model]
    args.optimizer_class = utils.module_to_dict(torch.optim)[args.optimizer]
    args.dataset_class = utils.module_to_dict(datasets)[args.dataset]

    return args


def initialize_distributed(args):
    # Manually set the device ids.
    torch.cuda.set_device(args.rank % torch.cuda.device_count())

    # Call the init process
    if args.world_size > 1:
        init_method = 'env://'
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=init_method)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_train_and_valid_data_loaders(block, args):
    transform = data_transforms.Compose([
        # geometric augmentation
        data_transforms.NumpyToPILImage(),
        data_transforms.RandomTranslate2D(max_displ_factor=0.05),
        data_transforms.RandomRotate2D(base_angle=17, delta_angle=5),
        data_transforms.RandomScaledCrop2D(crop_height=args.crop_size[0],
                                           crop_width=args.crop_size[1], min_crop_ratio=0.8),
        data_transforms.RandomVerticalFlip(prob=0.5),
        data_transforms.RandomHorizontalFlip(prob=0.5),
        data_transforms.PILImageToNumpy(),
        # photometric augmentation
        data_transforms.RandomGamma(gamma_low=0.9, gamma_high=1.1),
        data_transforms.RandomBrightness(brightness_factor=0.1),
        data_transforms.RandomColorOrder(prob=0.5),
        data_transforms.RandomContrast(contrast_low=-0.1, contrast_high=0.1),
        data_transforms.RandomSaturation(saturation_low=-0.1, saturation_high=0.1)
    ])

    if args.skip_aug:
        transform = data_transforms.Compose([
            # geometric augmentation
            data_transforms.NumpyToPILImage(),
            data_transforms.RandomCrop2D(crop_height=args.crop_size[0],
                                         crop_width=args.crop_size[1]),
            data_transforms.RandomVerticalFlip(prob=0.5),
            data_transforms.RandomHorizontalFlip(prob=0.5),
            data_transforms.PILImageToNumpy()
        ])

    # training dataloader
    tkwargs = {'batch_size': args.batch_size,
               'num_workers': args.workers,
               'pin_memory': True, 'drop_last': True}
    step_size = args.step_size if args.step_size > 0 else (args.num_interp + 1)
    train_dataset = args.dataset_class(args=args, root=args.train_file, num_interp=args.num_interp,
                                       sample_rate=args.sample_rate, step_size=step_size, is_training=True,
                                       transform=transform)

    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        shuffle=(train_sampler is None), **tkwargs)

    block.log('Number of Training Images: {}:({} mini-batches)'.format(
        step_size * len(train_loader.dataset), len(train_loader)))

    # validation dataloader
    vkwargs = {'batch_size': args.val_batch_size,
               'num_workers': args.workers,
               'pin_memory': True, 'drop_last': True}
    step_size = args.val_step_size if args.val_step_size > 0 else (args.val_num_interp + 1)

    val_dataset = args.dataset_class(args=args, root=args.val_file, num_interp=args.val_num_interp,
                                     sample_rate=args.val_sample_rate, step_size=step_size)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **vkwargs)

    block.log('Number of Validation Images: {}:({} mini-batches)'.format(
        step_size * len(val_loader.dataset), len(val_loader)))
    args.val_size = val_loader.dataset[0]['image'][0].shape[:2]

    return train_loader, train_sampler, val_loader


def load_model(model, optimizer, block, args):
    # trained weights
    checkpoint = torch.load(args.resume, map_location='cpu')

    # used for partial initialization
    input_dict = checkpoint['state_dict']
    curr_dict = model.state_dict()
    state_dict = input_dict.copy()
    for key in input_dict:
        if key not in curr_dict:
            print(key)
            continue
        if curr_dict[key].shape != input_dict[key].shape:
            state_dict.pop(key)
            print("key {} skipped because of size mismatch.".format(
                key))
    model.load_state_dict(state_dict, strict=False)
    if 'optimizer' in checkpoint and args.start_epoch < 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if args.start_epoch < 0:
        args.start_epoch = max(0, checkpoint['epoch'])
    block.log("Successfully loaded checkpoint (at epoch {})".format(
        checkpoint['epoch']))


def build_and_initialize_model_and_optimizer(block, args):
    model = args.network_class(args)
    block.log('Number of parameters: {val:,}'.format(val=
        sum([p.data.nelement()
             if p.requires_grad else 0 for p in model.parameters()])))

    block.log('Initializing CUDA')
    assert torch.cuda.is_available(), 'only GPUs support at the moment'
    model.cuda(torch.cuda.current_device())

    optimizer = args.optimizer_class(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    block.log("Attempting to Load checkpoint '{}'".format(args.resume))
    if args.resume and os.path.isfile(args.resume):
        load_model(model, optimizer, block, args)
    elif args.resume:
        block.log("No checkpoint found at '{}'".format(args.resume))
        exit(1)
    else:
        block.log("Random initialization, checkpoint not provided.")
        args.start_epoch = 0

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Run multi-process when it is needed.
    if args.world_size > 1:
        model = DistributedDataParallel(model)

    return model, optimizer


def get_learning_rate_scheduler(optimizer, block, args):
    block.log('Base leaning rate {}.'.format(args.lr))
    if args.lr_scheduler == 'ExponentialLR':
        block.log('Using exponential decay learning rate scheduler with '
                  '{} decay rate'.format(args.lr_gamma))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                              args.lr_gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        block.log('Using multi-step learning rate scheduler with {} gamma '
                   'and {} milestones.'.format(args.lr_gamma,
                                               args.lr_milestones))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'PolyLR':
        block.log('Using polynomial decay learning rate scheduler with {} gamma ' 
            'and {} milestones.'.format(args.lr_gamma,
                                               args.lr_milestones))

        lr_gamma = math.log(0.1) / math.log(1 - (args.lr_milestones[0] - 1e-6) / args.epochs)

        # Poly with lr_gamma until args.lr_milestones[0], then stepLR with factor of 0.1
        lambda_map = lambda epoch_index: math.pow(1 - epoch_index / args.epochs, lr_gamma) \
            if np.searchsorted(args.lr_milestones, epoch_index + 1) == 0 \
            else math.pow(10, -1 * np.searchsorted(args.lr_milestones, epoch_index + 1))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_map)

    else:
        raise NameError('Unknown {} learning rate scheduler'.format(
            args.lr_scheduler))

    return lr_scheduler


def forward_only(inputs_gpu, targets_gpu, model):
    # Forward pass.
    losses, outputs, targets = model(inputs_gpu, targets_gpu)

    # Loss.
    for k in losses:
        losses[k] = losses[k].mean(dim=0)
    loss = losses['tot']

    return loss, outputs, targets


def calc_linf_grad_norm(args,parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = max(p.grad.data.abs().max() for p in parameters)
    max_norm_reduced = torch.cuda.FloatTensor([max_norm])
    if args.world_size > 1:
        torch.distributed.all_reduce(max_norm_reduced,
                                     op=torch.distributed.ReduceOp.MAX)
    return max_norm_reduced[0].item()


def train_step(batch_cpu, model, optimizer, block, args, print_linf_grad=False):
    # Move data to GPU.

    inputs = {k: [b.cuda() for b in batch_cpu[k]]
              for k in batch_cpu if k in ['image', 'fwd_mvec', 'bwd_mvec', 'depth']}
    tar_index = batch_cpu['tindex'].cuda()

    # Forward pass.
    loss, outputs, targets = forward_only(inputs, tar_index, model)

    # Backward and SGP steps.
    optimizer.zero_grad()
    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    # Calculate and print norm infinity of the gradients.
    if print_linf_grad:
        block.log('gradients Linf: {:0.3f}'.format(calc_linf_grad_norm(args,
            model.parameters())))

    # Clip gradients by value.
    if args.clip_gradients > 0:
        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_gradients)

    optimizer.step()

    return loss, outputs, targets


def evaluate_epoch(model, val_loader, block, args, epoch=0):
    # Because train and val number of frame interpolate could be different.
    if args.val_num_interp != args.num_interp:
        model_ = model
        if args.world_size > 1:
            model_ = model.module
        model_.tlinespace = torch.linspace(
            0, 1, 2 + args.val_num_interp).float().cuda()

    # calculate validation loss, create videos, or dump predicted frames
    v_psnr, v_ssim, v_ie, loss_values = evaluate(args, val_loader, model, args.val_num_interp, epoch, block)

    if args.val_num_interp != args.num_interp:
        model_ = model
        if args.world_size > 1:
            model_ = model.module
        model_.tlinespace = torch.linspace(0, 1,
                                           2 + args.num_interp).float().cuda()
    # Move back the model to train mode.
    model.train()

    return v_psnr, v_ssim, v_ie, loss_values


def write_summary(global_index, learning_rate, t_loss,
                  v_loss, v_psnr, v_ssim, v_ie, args):
    # Write to tensorboard.
    if args.rank == 0:
        args.logger.add_scalar("lr", learning_rate, global_index)
        args.logger.add_scalars("Loss",
                                {'trainLoss': t_loss, 'valLoss': v_loss},
                                global_index)
        args.logger.add_scalar("PSNR", v_psnr, global_index)
        args.logger.add_scalar("SSIM", v_ssim, global_index)
        args.logger.add_scalar("RMS", v_ie, global_index)


def train_epoch(epoch, args, model, optimizer, lr_scheduler,
                train_sampler, train_loader,
                v_psnr, v_ssim, v_ie, v_loss, block):
    # Average loss calculator.
    loss_values = utils.AverageMeter()

    # Advance Learning rate.
    lr_scheduler.step()

    # This will ensure the data is shuffled each epoch.
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    # Get number of batches in one epoch.
    num_batches = len(train_loader) if args.train_n_batches < 0 \
        else args.train_n_batches

    global_index = 0
    for i, batch in enumerate(train_loader):

        # Set global index.
        global_index = epoch * num_batches + i

        # Move one step.
        loss, outputs, _ = train_step(
            batch, model, optimizer, block, args,
            ((global_index + 1) % args.print_freq == 0))

        # Update the loss accumulator.
        loss_values.update(loss.data.item(), outputs.size(0))

        # Summary writer.
        if (global_index + 1) % args.print_freq == 0:

            # Reduce the loss.
            if args.world_size > 1:
                t_loss_gpu = torch.Tensor([loss_values.val]).cuda()
                torch.distributed.all_reduce(t_loss_gpu)
                t_loss = t_loss_gpu.item() / args.world_size
            else:
                t_loss = loss_values.val

            # Write to tensorboard.
            write_summary(global_index, lr_scheduler.get_lr()[0], t_loss,
                          v_loss, v_psnr, v_ssim, v_ie, args)

            # And reset the loss accumulator.
            loss_values.reset()

            # Print some output.
            dict2print = {'iter': global_index,
                          'epoch': str(epoch) + '/' + str(args.epochs),
                          'batch': str(i + 1) + '/' + str(num_batches)}
            str2print = ' '.join(key + " : " + str(dict2print[key])
                                 for key in dict2print)
            str2print += ' trainLoss:' + ' %1.3f' % t_loss
            str2print += ' valLoss' + ' %1.3f' % v_loss
            str2print += ' valPSNR' + ' %1.3f' % v_psnr
            str2print += ' lr:' + ' %1.6f' % (lr_scheduler.get_lr()[0])
            block.log(str2print)

        # Break the training loop if we have reached the maximum number of batches.
        if (i + 1) >= num_batches:
            break
    return global_index


def save_model(model, optimizer, epoch, global_index, max_psnr, block, args):
    # Write on rank zero only
    if args.rank == 0:
        if args.world_size > 1:
            model_ = model.module
        else:
            model_ = model
        state_dict = model_.state_dict()
        tmp_keys = state_dict.copy()
        for k in state_dict:
            [tmp_keys.pop(k) if (k in tmp_keys and ikey in k)
             else None for ikey in model_.ignore_keys]
        state_dict = tmp_keys.copy()
        # save checkpoint
        model_optim_state = {'epoch': epoch,
                             'arch': args.model,
                             'state_dict': state_dict,
                             'optimizer': optimizer.state_dict(),
                             }
        model_name = os.path.join(
            args.save_root, '_ckpt_epoch_%03d_iter_%07d_psnr_%1.2f.pt.tar' % (
                epoch, global_index, max_psnr))
        torch.save(model_optim_state, model_name)
        block.log('saved model {}'.format(model_name))

        return model_name


def train(model, optimizer, lr_scheduler, train_loader,
          train_sampler, val_loader, block, args):
    # Set the model to train mode.
    model.train()

    # Keep track of maximum PSNR.
    max_psnr = -1

    # Perform an initial evaluation.
    if args.initial_eval:
        block.log('Initial evaluation.')

        v_psnr, v_ssim, v_ie, v_loss = evaluate_epoch(model, val_loader, block, args, args.start_epoch)
    else:
        v_psnr, v_ssim, v_ie, v_loss = 20.0, 0.5, 15.0, 0.0

    for epoch in range(args.start_epoch, args.epochs):

        # Train for an epoch.
        global_index = train_epoch(epoch, args, model, optimizer, lr_scheduler,
                                   train_sampler, train_loader, v_psnr, v_ssim, v_ie, v_loss, block)

        if (epoch + 1) % args.save_freq == 0:
            v_psnr, v_ssim, v_ie, v_loss = evaluate_epoch(model, val_loader, block, args, epoch + 1)
            if v_psnr > max_psnr:
                max_psnr = v_psnr
                save_model(model, optimizer, epoch + 1, global_index,
                           max_psnr, block, args)

    return 0


def main():
    # Parse the args.
    with utils.TimerBlock("\nParsing Arguments") as block:
        args = parse_and_set_args(block)

    # Initialize torch.distributed.
    with utils.TimerBlock("Initializing Distributed"):
        initialize_distributed(args)

    # Set all random seed for reproducibility.
    with utils.TimerBlock("Setting Random Seed"):
        set_random_seed(args.seed)

    # Train and validation data loaders.
    with utils.TimerBlock("Building {} Dataset".format(args.dataset)) as block:
        train_loader, train_sampler, val_loader = get_train_and_valid_data_loaders(block, args)

    # Build the model and optimizer.
    with utils.TimerBlock("Building {} Model and {} Optimizer".format(
            args.model, args.optimizer_class.__name__)) as block:
        model, optimizer = build_and_initialize_model_and_optimizer(block, args)

    # Learning rate scheduler.
    with utils.TimerBlock("Building {} Learning Rate Scheduler".format(
            args.optimizer)) as block:
        lr_scheduler = get_learning_rate_scheduler(optimizer, block, args)

    # Set the tf writer on rank 0.
    with utils.TimerBlock("Creating Tensorboard Writers"):
        if args.rank == 0:
            args.logger = SummaryWriter(log_dir=args.save_root)

    with utils.TimerBlock("Training Model") as block:
        train(model, optimizer, lr_scheduler, train_loader,
              train_sampler, val_loader, block, args)

    return 0


if __name__ == '__main__':
    main()

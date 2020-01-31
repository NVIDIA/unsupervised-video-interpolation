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
import random
from PIL import Image
import numpy as np
from torchvision.transforms import functional as transf

"""
Compose for Multiple Arguments
"""


class Compose(object):
    """Custom class to serialise transformations that
    accept multiple input arguments

    Args:
        transforms (list of ``Transform`` objects): list of custom transforms to compose

    Example:
        composed_transf = data_transforms.Compose(
            [NumpyToPILImage(),
             RandomScaledCrop2D(crop_height=384, crop_width=384, min_crop_ratio=0.8),
             PILImageToNumpy(),
             RandomReverseSequence(),
             RandomBrightness(brightness_factor=0.1)
        ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs, targets):
        for transform in self.transforms:
            inputs, targets = transform(inputs, targets)
        return inputs, targets


"""
Image Type Conversion
"""


class NumpyToPILImage(object):
    """Convert numpy array to an instance of PIL Image, so we can use
    geometric transformations already available in torchvision.transforms.functional.*.
    """

    def __call__(self, inputs, targets):
        inputs = [Image.fromarray(np.clip(im, 0, 255)) for im in inputs]
        targets = [Image.fromarray(np.clip(im, 0, 255)) for im in targets]
        return inputs, targets


class PILImageToNumpy(object):
    """Convert PIL Image to a numpy array at the end of geometric transformations.
    Note. All photometric transformations currently work on numpy arrays, because for some
    transformations, there is an implementation mis-match between torchvision and the ones defined
    in flownet2 (Caffe: https://github.com/lmb-freiburg/flownet2), which they are derived/inspired from.
    """

    def __call__(self, inputs, targets):
        inputs = [np.array(im) for im in inputs]
        targets = [np.array(im) for im in targets]
        return inputs, targets


""" 
Geometric Augmentation 
"""


class RandomRotate2D(object):
    """Apply random 2D in-plane rotation of on input and target image sequences.
    For video interpolation or optical flow studies, we also add a small
    offset rotation to each image in the sequence ranging from [-delta, delta] degrees
    in a linear fashion, such that networks can learn to recover the added fake rotation.
    """
    def __init__(self, base_angle=20, delta_angle=0, resample=Image.BILINEAR):
        self.base_angle = base_angle
        self.delta_angle = delta_angle
        self.resample = resample

    def __call__(self, inputs, targets):
        base = random.uniform(-self.base_angle, self.base_angle)
        delta = random.uniform(-self.delta_angle, self.delta_angle)
        resample = self.resample

        inputs[0] = transf.rotate(inputs[0], angle=(base - delta / 2.), resample=resample)
        inputs[-1] = transf.rotate(inputs[1], angle=(base + delta / 2.), resample=resample)

        # Apply linearly varying offset to targets
        # calculate offset ~ (-delta/2., delta/2.)
        tlinspace = np.linspace(-1, 1, len(targets) + 2)
        for i, image in enumerate(targets):
            offset = tlinspace[i + 1] * delta / 2.
            targets[i] = transf.rotate(image, angle=(base + offset), resample=resample)

        return inputs, targets


class RandomTranslate2D(object):
    """Apply random 2D translation on input and target image sequences.
    For video interpolation or optical flow studies, we also add a small
    offset translation to each image in the sequence ranging from [-delta, delta] pixel displacements
    in a linear fashion, such that networks can learn to recover the added fake translation.
    """
    def __init__(self, max_displ_factor=0.05, resample=Image.NEAREST):
        self.max_displ_factor = max_displ_factor
        self.resample = resample

    def __call__(self, inputs, targets):
        # h, w, _ = inputs[0].shape
        w, h = inputs[0].size
        max_displ_factor = self.max_displ_factor
        resample = self.resample

        # Sample a displacement in [-max_displ, max_displ] for both height and width
        max_width_displ = int(w * max_displ_factor)
        wd = random.randint(-max_width_displ, max_width_displ)

        max_height_displ = int(h * max_displ_factor)
        hd = random.randint(-max_height_displ, max_height_displ)

        inputs[0] = transf.affine(inputs[0], angle=0, translate=(wd, hd), scale=1, shear=0, resample=resample)
        inputs[-1] = transf.affine(inputs[-1], angle=0, translate=(-wd, -hd), scale=1, shear=0, resample=resample)

        # Apply linearly varying offset to targets
        # calculate offset ~ (-{w|h}_delta, {w|h}_delta})
        tlinspace = -1 * np.linspace(-1, 1, len(targets) + 2)
        for i, image in enumerate(targets):
            wo, ho = tlinspace[i + 1] * wd, tlinspace[i + 1] * hd
            targets[i] = transf.affine(image, angle=0, translate=(wo, ho), scale=1, shear=0, resample=resample)

        return inputs, targets


class RandomCrop2D(object):
    """A simple random 3D crop with a provided crop_size.
    """
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, inputs, targets):
        width, height = inputs[0].size
        crop_width, crop_height = self.crop_width, self.crop_height

        # sample crop indices
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)

        for i, image in enumerate(inputs):
            inputs[i] = transf.crop(image, top, left, crop_height, crop_width)

        for i, image in enumerate(targets):
            targets[i] = transf.crop(image, top, left, crop_height, crop_width)

        return inputs, targets


class RandomScaledCrop2D(object):
    """Apply random 2D crop followed by a scale operation.
    Note to simulate a simple crop, set
        ``min_crop_ratio=min(crop_height,crop_width)/min(height, width)``.
    We basically, first, crop the original image with a size larger or smaller than
    the desired crop size. We then scale the images to the desired crop_size.
    So, in a way, this transformation encapsulates two augmentations: scale + crop.
    """

    def __init__(self, crop_height, crop_width, min_crop_ratio=0.6, resample=Image.BILINEAR):
        # Aspect ratio inherited from (crop_height, crop_width)
        self.crop_aspect = crop_height / crop_width
        self.crop_shape = (crop_height, crop_width)
        self.min_crop_ratio = min_crop_ratio
        self.resample = resample

    def __call__(self, inputs, targets):
        # height, width, _ = inputs[0].shape
        width, height = inputs[0].size
        crop_aspect = self.crop_aspect
        crop_shape = self.crop_shape
        resample = self.resample
        min_crop_ratio = self.min_crop_ratio

        source_aspect = height / width

        # sample a crop factor in [min_crop_ratio, 1.)
        crop_ratio = random.uniform(min_crop_ratio, 1.0)

        # Preserve aspect ratio provided by (crop_height, crop_width)
        # Calculate crop height and with, apply crop_ratio along the min(height,width)'s axis
        if crop_aspect < source_aspect:
            cwidth = int(width * crop_ratio)
            cheight = int(cwidth * crop_aspect)
        else:
            cheight = int(height * crop_ratio)
            cwidth = int(cheight / crop_aspect)

        # Avoid bilinear re-sampling crop_size == full_size
        if cheight == cwidth and cwidth == width:
            return inputs, targets

        # sample crop indices
        left = random.randint(0, width - cwidth)
        top = random.randint(0, height - cheight)

        for i, image in enumerate(inputs):
            inputs[i] = transf.resized_crop(inputs[i], top, left, cheight, cwidth, crop_shape, interpolation=resample)
        for i, image in enumerate(targets):
            targets[i] = transf.resized_crop(targets[i], top, left, cheight, cwidth, crop_shape, interpolation=resample)

        return inputs, targets


class RandomHorizontalFlip(object):
    """Apply a random horizontal flip."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs, targets):
        #
        if random.random() < self.prob:
            return inputs, targets

        # Apply a horizontal flip
        for i, image in enumerate(inputs):
            inputs[i] = transf.hflip(image)
        for i, image in enumerate(targets):
            targets[i] = transf.hflip(image)

        return inputs, targets


class RandomVerticalFlip(object):
    """Apply a random vertical flip."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs, targets):
        #
        if random.random() < self.prob:
            return inputs, targets

        # Apply a vertical flip
        for i, image in enumerate(inputs):
            inputs[i] = transf.vflip(image)
        for i, image in enumerate(targets):
            targets[i] = transf.vflip(image)

        return inputs, targets


class RandomReverseSequence(object):
    """Randomly reverse the order of inputs, and targets"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs, targets):
        if random.random() < self.prob:
            return inputs, targets

        # Reverse sequence
        inputs = inputs[::-1]
        targets = targets[::-1]

        return inputs, targets


"""
Photometric Augmentation 
"""


class RandomGamma(object):
    """Apply a gamma transformation, with gamma factor of (gamma_low, anf gamma_high)"""

    def __init__(self, gamma_low, gamma_high):
        self.gamma_low = gamma_low
        self.gamma_high = gamma_high

    def __call__(self, inputs, targets):
        gamma = random.uniform(self.gamma_low, self.gamma_high)

        if gamma == 1.0:
            return inputs, targets
        gamma_inv = 1. / gamma

        # Apply a gamma
        for i, image in enumerate(inputs):
            image = np.power(image / 255.0, gamma_inv) * 255.0
            inputs[i] = np.clip(image, 0., 255.)

        for i, image in enumerate(targets):
            image = np.power(image / 255.0, gamma_inv) * 255.0
            targets[i] = np.clip(image, 0., 255.)

        return inputs, targets


class RandomBrightness(object):
    """Apply a random brightness to each channel in the image.
    An implementation that is quite distinct from torchvision.
    """

    def __init__(self, brightness_factor=0.1):
        self.brightness_factor = brightness_factor

    def __call__(self, inputs, targets):
        brighness_factor = [1 + random.uniform(-self.brightness_factor, self.brightness_factor) for _ in range(3)]
        brighness_factor = np.array(brighness_factor)

        # Apply a brightness
        for i, image in enumerate(inputs):
            image = image * brighness_factor
            inputs[i] = np.clip(image, 0., 255.)

        for i, image in enumerate(targets):
            image = image * brighness_factor
            targets[i] = np.clip(image, 0., 255.)

        return inputs, targets


class RandomColorOrder(object):
    """Randomly re-order the channels of images.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs, targets):
        if random.random() < self.prob:
            return inputs, targets

        new_order = np.random.permutation(3)

        # Apply a brightness
        for i, image in enumerate(inputs):
            inputs[i] = image[..., new_order]
        for i, image in enumerate(targets):
            targets[i] = image[..., new_order]

        return inputs, targets


class RandomContrast(object):
    """Apply a random contrast in the range (contrast_low, contrast_high) to all channels.
    An implementation that is quite distinct from torchvision.
    """

    def __init__(self, contrast_low, contrast_high):
        self.contrast_low = contrast_low
        self.contrast_high = contrast_high

    def __call__(self, inputs, targets):
        contrast = 1 + random.uniform(self.contrast_low, self.contrast_high)

        # Apply a contrast
        for i, image in enumerate(inputs):
            gray_img = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
            tmp_img = np.ones_like(image) * gray_img.mean()
            image = image * contrast + (1 - contrast) * tmp_img
            inputs[i] = np.clip(image, 0, 255)

        for i, image in enumerate(targets):
            gray_img = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
            tmp_img = np.ones_like(image) * gray_img.mean()
            image = image * contrast + (1 - contrast) * tmp_img
            targets[i] = np.clip(image, 0, 255)

        return inputs, targets


class RandomSaturation(object):
    """Apply a random saturation in the range (saturation_low, saturation_high) to all channels.
    An implementation that is quite distinct from torchvision.
    """

    def __init__(self, saturation_low, saturation_high):
        self.saturation_low = saturation_low
        self.saturation_high = saturation_high

    def __call__(self, inputs, targets):
        saturation = 1 + random.uniform(self.saturation_low, self.saturation_high)
        if saturation == 1.0:
            return inputs, targets

        # Apply a saturation
        for i, image in enumerate(inputs):
            gray_img = image[..., 0] * 0.299 + image[..., 1] * image[..., 2] * 0.114
            tmp_img = np.stack((gray_img, gray_img, gray_img), axis=2)
            image = image * saturation + (1 - saturation) * tmp_img
            inputs[i] = np.clip(image, 0, 255)

        for i, image in enumerate(targets):
            gray_img = image[..., 0] * 0.299 + image[..., 1] * image[..., 2] * 0.114
            tmp_img = np.stack((gray_img, gray_img, gray_img), axis=2)
            image = image * saturation + (1 - saturation) * tmp_img
            targets[i] = np.clip(image, 0, 255)

        return inputs, targets

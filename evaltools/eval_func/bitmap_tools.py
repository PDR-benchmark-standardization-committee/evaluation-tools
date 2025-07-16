#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from PIL import Image
import numpy as np


def load_bitmap_to_ndarray(filename):
    # load image as PIL Image Object
    image = Image.open(filename).convert('L')
    # convert to ndarray and scaling 0-1
    array = np.array(image) / 255.0

    return np.flipud(array).T


def save_bitmap_from_ndarray(ndarray, filename):
    # Scale 2D ndarray to the range [0, 255]
    scaled_array = (ndarray * 255).astype(np.uint8)
    # Convert a 2D ndarray to a PIL Image object
    image = Image.fromarray(scaled_array)
    # save
    image.save(filename)

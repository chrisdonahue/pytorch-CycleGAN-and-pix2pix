"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
  -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
  -- <__init__>: Initialize this dataset class.
  -- <__getitem__>: Return a data point and its metadata information.
  -- <__len__>: Return the number of images.
"""
import glob
import os
import random
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image

def _random_slice(spec, slice_len=256):
  spec_len = spec.shape[0]
  center_idx = random.randint(0, spec_len - 1)
  pad_amt = slice_len // 2
  spec_padded = np.pad(spec, [[pad_amt, pad_amt], [0, 256 - 229], [0, 0]], 'reflect')
  return spec_padded[center_idx:center_idx+slice_len]


class PianoiseDataset(BaseDataset):
  """A template dataset class for you to implement custom datasets."""
  @staticmethod
  def modify_commandline_options(parser, is_train):
    """Add new dataset-specific options, and rewrite default values for existing options.

    Parameters:
      parser      -- original option parser
      is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

    Returns:
      the modified parser.
    """
    parser.add_argument('--a_mean_fp', type=str)
    parser.add_argument('--a_std_fp', type=str)
    parser.add_argument('--b_mean_fp', type=str)
    parser.add_argument('--b_std_fp', type=str)
    parser.add_argument('--num_std_devs', type=float)
    return parser

  def __init__(self, opt):
    """Initialize this dataset class.

    Parameters:
      opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

    A few things can be done here.
    - save the options (have been done in BaseDataset)
    - get image paths and meta information of the dataset.
    - define the image transformation.
    """
    # save the option and dataset root
    BaseDataset.__init__(self, opt)

    self.A_paths = sorted(glob.glob(os.path.join(opt.dataroot, 'ddc', opt.phase, '*.npy')))
    self.B_paths = sorted(glob.glob(os.path.join(opt.dataroot, 'maestro', opt.phase, '*.npy')))
    try:
      self.A_paths = self.A_paths[:opt.max_dataset_size]
      self.B_paths = self.B_paths[:opt.max_dataset_size]
    except:
      pass

    self.a_mean = np.load(opt.a_mean_fp)
    self.a_std = np.load(opt.a_std_fp)
    self.b_mean = np.load(opt.b_mean_fp)
    self.b_std = np.load(opt.b_std_fp)

    self.num_std_devs = opt.num_std_devs

  def __getitem__(self, index):
    """Return a data point and its metadata information.

    Parameters:
      index -- a random integer for data indexing

    Returns:
      a dictionary of data with their names. It usually contains the data itself and its metadata information.

    Step 1: get a random image path: e.g., path = self.image_paths[index]
    Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
    Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
    Step 4: return a data point as a dictionary.
    """
    a_path = random.choice(self.A_paths)
    b_path = random.choice(self.B_paths)

    a = np.load(a_path)
    b = np.load(b_path)

    a -= self.a_mean
    a /= self.a_std
    a /= self.num_std_devs
    a = np.clip(a, -1., 1.)

    b -= self.b_mean
    b /= self.b_std
    b /= self.num_std_devs
    b = np.clip(b, -1., 1.)

    a = _random_slice(a)
    b = _random_slice(b)

    a = np.transpose(a, [2, 0, 1])
    b = np.transpose(b, [2, 0, 1])

    a = torch.tensor(a)
    b = torch.tensor(b)

    assert a.shape == (1, 256, 256)
    assert b.shape == (1, 256, 256)

    return {'A': a, 'B': b, 'A_paths': a_path, 'B_paths': b_path}

  def __len__(self):
    """Return the total number of images."""
    AVERAGE_SONG_LENGTH = 180.
    WINDOW_LENGTH_SECONDS = 8.192
    return max(len(self.A_paths), len(self.B_paths)) * int(AVERAGE_SONG_LENGTH / WINDOW_LENGTH_SECONDS)

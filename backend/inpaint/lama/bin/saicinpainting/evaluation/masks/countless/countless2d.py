from __future__ import print_function, division

"""
COUNTLESS performance test in Python.

python countless2d.py ./images/NAMEOFIMAGE
"""

import six
from six.moves import range
from collections import defaultdict
from functools import reduce
import operator 
import io
import os
from PIL import Image
import math
import numpy as np
import random
import sys
import time
from tqdm import tqdm
from scipy import ndimage

def simplest_countless(data):
  """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
  sections = []
  
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections

  ab = a * (a == b) # PICK(A,B)
  ac = a * (a == c) # PICK(A,C)
  bc = b * (b == c) # PICK(B,C)

  a = ab | ac | bc # Bitwise OR, safe b/c non-matches are zeroed
  
  return a + (a == 0) * d # AB || AC || BC || D

def quick_countless(data):
  """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
  sections = []
  
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections

  ab_ac = a * ((a == b) | (a == c)) # PICK(A,B) || PICK(A,C) w/ optimization
  bc = b * (b == c) # PICK(B,C)

  a = ab_ac | bc # (PICK(A,B) || PICK(A,C)) or PICK(B,C)
  return a + (a == 0) * d # AB || AC || BC || D

def quickest_countless(data):
  """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
  sections = []
  
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections

  ab_ac = a * ((a == b) | (a == c)) # PICK(A,B) || PICK(A,C) w/ optimization
  ab_ac |= b * (b == c) # PICK(B,C)
  return ab_ac + (ab_ac == 0) * d # AB || AC || BC || D

def quick_countless_xor(data):
  """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
  sections = []
  
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections

  ab = a ^ (a ^ b) # a or b
  ab += (ab != a) * ((ab ^ (ab ^ c)) - b) # b or c
  ab += (ab == c) * ((ab ^ (ab ^ d)) - c) # c or d
  return ab

def stippled_countless(data):
  """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm
  that treats zero as "background" and inflates lone
  pixels.
  
  data is a 2D numpy array with even dimensions.
  """
  sections = []
  
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections

  ab_ac = a * ((a == b) | (a == c)) # PICK(A,B) || PICK(A,C) w/ optimization
  ab_ac |= b * (b == c) # PICK(B,C)

  nonzero = a + (a == 0) * (b + (b == 0) * c)
  return ab_ac + (ab_ac == 0) * (d + (d == 0) * nonzero) # AB || AC || BC || D

def zero_corrected_countless(data):
  """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
  # allows us to prevent losing 1/2 a bit of information 
  # at the top end by using a bigger type. Without this 255 is handled incorrectly.
  data, upgraded = upgrade_type(data) 

  # offset from zero, raw countless doesn't handle 0 correctly
  # we'll remove the extra 1 at the end.
  data += 1 

  sections = []
  
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections

  ab = a * (a == b) # PICK(A,B)
  ac = a * (a == c) # PICK(A,C)
  bc = b * (b == c) # PICK(B,C)

  a = ab | ac | bc # Bitwise OR, safe b/c non-matches are zeroed
  
  result = a + (a == 0) * d - 1 # a or d - 1

  if upgraded:
    return downgrade_type(result)

  # only need to reset data if we weren't upgraded 
  # b/c no copy was made in that case
  data -= 1

  return result

def countless_extreme(data):
  nonzeros = np.count_nonzero(data)
  # print("nonzeros", nonzeros)

  N = reduce(operator.mul, data.shape)

  if nonzeros == N:
    print("quick")
    return quick_countless(data)
  elif np.count_nonzero(data + 1) == N:
    print("quick")
    # print("upper", nonzeros)
    return quick_countless(data)
  else:
    return countless(data)


def countless(data):
  """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
  # allows us to prevent losing 1/2 a bit of information 
  # at the top end by using a bigger type. Without this 255 is handled incorrectly.
  data, upgraded = upgrade_type(data) 

  # offset from zero, raw countless doesn't handle 0 correctly
  # we'll remove the extra 1 at the end.
  data += 1 

  sections = []
  
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections

  ab_ac = a * ((a == b) | (a == c)) # PICK(A,B) || PICK(A,C) w/ optimization
  ab_ac |= b * (b == c) # PICK(B,C)
  result = ab_ac + (ab_ac == 0) * d - 1 # (matches or d) - 1

  if upgraded:
    return downgrade_type(result)

  # only need to reset data if we weren't upgraded 
  # b/c no copy was made in that case
  data -= 1

  return result

def upgrade_type(arr):
  dtype = arr.dtype

  if dtype == np.uint8:
    return arr.astype(np.uint16), True
  elif dtype == np.uint16:
    return arr.astype(np.uint32), True
  elif dtype == np.uint32:
    return arr.astype(np.uint64), True

  return arr, False
  
def downgrade_type(arr):
  dtype = arr.dtype

  if dtype == np.uint64:
    return arr.astype(np.uint32)
  elif dtype == np.uint32:
    return arr.astype(np.uint16)
  elif dtype == np.uint16:
    return arr.astype(np.uint8)
  
  return arr

def odd_to_even(image):
  """
  To facilitate 2x2 downsampling segmentation, change an odd sized image into an even sized one.
  Works by mirroring the starting 1 pixel edge of the image on odd shaped sides.

  e.g. turn a 3x3x5 image into a 4x4x5 (the x and y are what are getting downsampled)
  
  For example: [ 3, 2, 4 ] => [ 3, 3, 2, 4 ] which is now easy to downsample.

  """
  shape = np.array(image.shape)

  offset = (shape % 2)[:2] # x,y offset
  
  # detect if we're dealing with an even
  # image. if so it's fine, just return.
  if not np.any(offset): 
    return image

  oddshape = image.shape[:2] + offset
  oddshape = np.append(oddshape, shape[2:])
  oddshape = oddshape.astype(int)

  newimg = np.empty(shape=oddshape, dtype=image.dtype)

  ox,oy = offset
  sx,sy = oddshape

  newimg[0,0] = image[0,0] # corner
  newimg[ox:sx,0] = image[:,0] # x axis line
  newimg[0,oy:sy] = image[0,:] # y axis line 

  return newimg

def counting(array):
    factor = (2, 2, 1)
    shape = array.shape

    while len(shape) < 4:
      array = np.expand_dims(array, axis=-1)
      shape = array.shape

    output_shape = tuple(int(math.ceil(s / f)) for s, f in zip(shape, factor))
    output = np.zeros(output_shape, dtype=array.dtype)

    for chan in range(0, shape[3]):
      for z in range(0, shape[2]):
        for x in range(0, shape[0], 2):
          for y in range(0, shape[1], 2):
            block = array[ x:x+2, y:y+2, z, chan ] # 2x2 block

            hashtable = defaultdict(int)
            for subx, suby in np.ndindex(block.shape[0], block.shape[1]):
              hashtable[block[subx, suby]] += 1

            best = (0, 0)
            for segid, val in six.iteritems(hashtable):
              if best[1] < val:
                best = (segid, val)

            output[ x // 2, y // 2, chan ] = best[0]
    
    return output

def ndzoom(array):
    if len(array.shape) == 3:
      ratio = ( 1 / 2.0, 1 / 2.0, 1.0 )
    else:
      ratio = ( 1 / 2.0, 1 / 2.0)
    return ndimage.interpolation.zoom(array, ratio, order=1)

def countless_if(array):
    factor = (2, 2, 1)
    shape = array.shape

    if len(shape) < 3:
      array = array[ :,:, np.newaxis ]
      shape = array.shape

    output_shape = tuple(int(math.ceil(s / f)) for s, f in zip(shape, factor))
    output = np.zeros(output_shape, dtype=array.dtype)

    for chan in range(0, shape[2]):
      for x in range(0, shape[0], 2):
        for y in range(0, shape[1], 2):
          block = array[ x:x+2, y:y+2, chan ] # 2x2 block

          if block[0,0] == block[1,0]:
            pick = block[0,0]
          elif block[0,0] == block[0,1]:
            pick = block[0,0]
          elif block[1,0] == block[0,1]:
            pick = block[1,0]
          else:
            pick = block[1,1]

          output[ x // 2, y // 2, chan ] = pick
    
    return np.squeeze(output)

def downsample_with_averaging(array):
  """
  Downsample x by factor using averaging.

  @return: The downsampled array, of the same type as x.
  """

  if len(array.shape) == 3:
    factor = (2,2,1)
  else:
    factor = (2,2)
  
  if np.array_equal(factor[:3], np.array([1,1,1])):
    return array

  output_shape = tuple(int(math.ceil(s / f)) for s, f in zip(array.shape, factor))
  temp = np.zeros(output_shape, float)
  counts = np.zeros(output_shape, np.int)
  for offset in np.ndindex(factor):
      part = array[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
      indexing_expr = tuple(np.s_[:s] for s in part.shape)
      temp[indexing_expr] += part
      counts[indexing_expr] += 1
  return np.cast[array.dtype](temp / counts)

def downsample_with_max_pooling(array):

  factor = (2,2)

  if np.all(np.array(factor, int) == 1):
      return array

  sections = []

  for offset in np.ndindex(factor):
    part = array[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  output = sections[0].copy()

  for section in sections[1:]:
    np.maximum(output, section, output)

  return output

def striding(array): 
  """Downsample x by factor using striding.

  @return: The downsampled array, of the same type as x.
  """
  factor = (2,2)
  if np.all(np.array(factor, int) == 1):
    return array
  return array[tuple(np.s_[::f] for f in factor)]

def benchmark():
  filename = sys.argv[1]
  img = Image.open(filename)
  data = np.array(img.getdata(), dtype=np.uint8)

  if len(data.shape) == 1:
    n_channels = 1
    reshape = (img.height, img.width)
  else:
    n_channels = min(data.shape[1], 3)
    data = data[:, :n_channels]
    reshape = (img.height, img.width, n_channels)

  data = data.reshape(reshape).astype(np.uint8)

  methods = [
    simplest_countless,
    quick_countless,
    quick_countless_xor,
    quickest_countless,
    stippled_countless,
    zero_corrected_countless,
    countless,
    downsample_with_averaging,
    downsample_with_max_pooling,
    ndzoom,
    striding,
    # countless_if,
    # counting,
  ]

  formats = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
  }

  if not os.path.exists('./results'):
    os.mkdir('./results')

  N = 500
  img_size = float(img.width * img.height) / 1024.0 / 1024.0
  print("N = %d, %dx%d (%.2f MPx) %d chan, %s" % (N, img.width, img.height, img_size, n_channels, filename))
  print("Algorithm\tMPx/sec\tMB/sec\tSec")
  for fn in methods:
    print(fn.__name__, end='')
    sys.stdout.flush()

    start = time.time()
    # tqdm is here to show you what's going on the first time you run it.
    # Feel free to remove it to get slightly more accurate timing results.
    for _ in tqdm(range(N), desc=fn.__name__, disable=True):
      result = fn(data)
    end = time.time()
    print("\r", end='')

    total_time = (end - start)
    mpx = N * img_size / total_time
    mbytes = N * img_size * n_channels / total_time
    # Output in tab separated format to enable copy-paste into excel/numbers
    print("%s\t%.3f\t%.3f\t%.2f" % (fn.__name__, mpx, mbytes, total_time))
    outimg = Image.fromarray(np.squeeze(result), formats[n_channels])
    outimg.save('./results/{}.png'.format(fn.__name__, "PNG"))

if __name__ == '__main__':
  benchmark()


# Example results:
# N = 5, 1024x1024 (1.00 MPx) 1 chan, images/gray_segmentation.png
# Function                        MPx/sec   MB/sec     Sec
# simplest_countless              752.855   752.855    0.01
# quick_countless                 920.328   920.328    0.01
# zero_corrected_countless        534.143   534.143    0.01
# countless                       644.247   644.247    0.01
# downsample_with_averaging       372.575   372.575    0.01
# downsample_with_max_pooling     974.060   974.060    0.01
# ndzoom                          137.517   137.517    0.04
# striding                      38550.588 38550.588    0.00
# countless_if                      4.377     4.377    1.14
# counting                          0.117     0.117   42.85

# Run without non-numpy implementations:
# N = 2000, 1024x1024 (1.00 MPx) 1 chan, images/gray_segmentation.png
# Algorithm                       MPx/sec   MB/sec     Sec
# simplest_countless              800.522   800.522    2.50
# quick_countless                 945.420   945.420    2.12
# quickest_countless              947.256   947.256    2.11
# stippled_countless              544.049   544.049    3.68
# zero_corrected_countless        575.310   575.310    3.48
# countless                       646.684   646.684    3.09
# downsample_with_averaging       385.132   385.132    5.19
# downsample_with_max_poolin      988.361   988.361    2.02
# ndzoom                          163.104   163.104   12.26
# striding                      81589.340 81589.340    0.02





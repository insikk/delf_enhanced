# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Matches two images using their DELF features.

The matching is done using feature-based nearest-neighbor search, followed by
geometric verification using RANSAC.

The DELF features can be extracted using the extract_features.py script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from delf import feature_io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import cKDTree
from scipy.misc import imresize

from PIL import Image
import io

from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import sys
import tensorflow as tf
from tensorflow.python.platform import app

import cv2

cmd_args = None

_DISTANCE_THRESHOLD = 0.8
IMAGE_SIZE = (16, 12)


def load_image_into_numpy_array(image):
    """
    Convert PIL image into 3d tensor. (RGB)
    
    Args:
        image: PIL.Image objet

    """    
    if image.mode == "P": # PNG palette mode
        image = image.convert('RGBA')
        # image.palette = None # PIL Bug Workaround

    (im_width, im_height) = image.size
    imgarray = np.asarray(image).reshape(
        (im_height, im_width, -1)).astype(np.uint8)

    # logger.info('image array mode: {}'.format(image.mode))
    # logger.info('image array interface shape: {}'.format(image.__array_interface__['shape']))
    # logger.info('imgarray shape: {}'.format(imgarray.shape))
    return imgarray[:, :, :3] # truncate alpha channel if exists. 

def read_image(image_path):
    with open(image_path, 'rb') as image_fp:
        image = Image.open(image_fp)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
    return image_np


def get_inliers(locations_1, descriptors_1, locations_2, descriptors_2):  
  
  num_features_1 = locations_1.shape[0]
  # tf.logging.info("Loaded image 1's %d features" % num_features_1)

  num_features_2 = locations_2.shape[0]
  # tf.logging.info("Loaded image 2's %d features" % num_features_2)

  # Find nearest-neighbor matches using a KD tree.
  d1_tree = cKDTree(descriptors_1)
  distances, indices = d1_tree.query(
      descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

  # Select feature locations for putative matches.
  locations_2_to_use = np.array([
      locations_2[i,] for i in range(num_features_2)
      if indices[i] != num_features_1
  ])
  locations_1_to_use = np.array([
      locations_1[indices[i],] for i in range(num_features_2)
      if indices[i] != num_features_1
  ])

  # Perform geometric verification using RANSAC.
  model_robust, inliers = ransac(
      (locations_1_to_use, locations_2_to_use),
      AffineTransform,
      min_samples=3,
      residual_threshold=20,
      max_trials=1000)
  return inliers, locations_1_to_use, locations_2_to_use


def get_attention_image_byte(att_score):  
  attention_np = np.squeeze(att_score, (0, 3)).astype(np.uint8)

  im = Image.fromarray(np.dstack((attention_np, attention_np, attention_np)))
  buf = io.BytesIO()
  im.save(buf, 'PNG')
  return buf.getvalue()

    

def get_ransac_image_byte(img_1, locations_1, descriptors_1, img_2, locations_2, descriptors_2, save_path=None, use_opencv_match_vis=True):
  """
  Args:
      img_1: image bytes. JPEG, PNG
      img_2: image bytes. JPEG, PNG

  Return:
      ransacn result PNG image as byte
      score: number of matching inlier
  """

  # Convert image byte to 3 channel numpy array
  with Image.open(io.BytesIO(img_1)) as img:
    img_1 = load_image_into_numpy_array(img)
  with Image.open(io.BytesIO(img_2)) as img:
    img_2 = load_image_into_numpy_array(img)

  inliers, locations_1_to_use, locations_2_to_use = get_inliers(locations_1, descriptors_1, locations_2, descriptors_2)

  # Visualize correspondences, and save to file.
  fig, ax = plt.subplots(figsize=IMAGE_SIZE)
  inlier_idxs = np.nonzero(inliers)[0]
  score = sum(inliers)
  if score is None:
    score = 0
#   # For different size of image, transform img_1 to fit to img_2
#   print('img_1 shape', img_1.shape)
#   print('img_1 type', type(img_1))
#   print('img_2 shape', img_2.shape)

#   ratio = float(img_2.shape[1]) / img_1.shape[1]
#   print('ratio', ratio)

#   resize_img_1 = imresize(img_1, ratio, interp='bilinear', mode=None)
#   print('resize_img_1 shape', resize_img_1.shape)

  if use_opencv_match_vis:
    inlier_matches = []
    for idx in inlier_idxs:
        inlier_matches.append(cv2.DMatch(idx, idx, 0))
        
    kp1 =[]
    for point in locations_1_to_use:
        kp = cv2.KeyPoint(point[1], point[0], _size=1)
        kp1.append(kp)

    kp2 =[]
    for point in locations_2_to_use:
        kp = cv2.KeyPoint(point[1], point[0], _size=1)
        kp2.append(kp)


    ransac_img = cv2.drawMatches(img_1, kp1, img_2, kp2, inlier_matches, None, flags=0)
    ransac_img = cv2.cvtColor(ransac_img, cv2.COLOR_BGR2RGB)    
    image_byte = cv2.imencode('.png', ransac_img)[1].tostring()

  else:
    plot_matches(
        ax,
        img_1,
        img_2,
        locations_1_to_use,
        locations_2_to_use,
        np.column_stack((inlier_idxs, inlier_idxs)),
        matches_color='b')
    ax.axis('off')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())      
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches=extent, format='png')
    plt.close('all') # close resources. 
    image_byte = buf.getvalue()

  return image_byte, score


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Read features.
  locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(
      cmd_args.features_1_path)
  locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(
      cmd_args.features_2_path)

  img_1 = mpimg.imread(cmd_args.image_1_path)
  img_2 = mpimg.imread(cmd_args.image_2_path)
  get_ransac_image_byte(img_1, locations_1, descriptors_1, img_2, locations_2, descriptors_2, cmd_args.output_image)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--image_1_path',
      type=str,
      default='test_images/image_1.jpg',
      help="""
      Path to test image 1.
      """)
  parser.add_argument(
      '--image_2_path',
      type=str,
      default='test_images/image_2.jpg',
      help="""
      Path to test image 2.
      """)
  parser.add_argument(
      '--features_1_path',
      type=str,
      default='test_features/image_1.delf',
      help="""
      Path to DELF features from image 1.
      """)
  parser.add_argument(
      '--features_2_path',
      type=str,
      default='test_features/image_2.delf',
      help="""
      Path to DELF features from image 2.
      """)
  parser.add_argument(
      '--output_image',
      type=str,
      default='test_match.png',
      help="""
      Path where an image showing the matches will be saved.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)

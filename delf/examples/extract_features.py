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

"""Extracts DELF features from a list of images, saving them to file.

The images must be in JPG format. The program checks if descriptors already
exist, and skips computation for those.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from google.protobuf import text_format
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.python.platform import app
import time

from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io
from delf import feature_pb2

cmd_args = None

# Extension of feature files.
_DELF_EXT = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def _ReadImageList(list_path):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths


# TODO: add method for single inference. 
class InferenceHelper():
    def __init__(self, config_path):
        tf.logging.set_verbosity(tf.logging.INFO) 

        # Parse DelfConfig proto.
        config = delf_config_pb2.DelfConfig()
        self.config = config
        with tf.gfile.FastGFile(config_path, 'r') as f:
            text_format.Merge(f.read(), config)


        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            self.byte_value = tf.placeholder(tf.string)
            self.image_tf = tf.image.decode_jpeg(self.byte_value, channels=3)

            self.input_image = tf.placeholder(tf.uint8, [None, None, 3])

            self.input_score_threshold = tf.placeholder(tf.float32)
            self.input_image_scales = tf.placeholder(tf.float32, [None])
            self.input_max_feature_num = tf.placeholder(tf.int32)            

            model_fn = feature_extractor.BuildModel('resnet_v1_50/block3', 'softplus', 'use_l2_normalized_feature', 1)
            

            boxes, self.feature_scales, features, scores, self.attention = feature_extractor.ExtractKeypointDescriptor(
                self.input_image,
                layer_name='resnet_v1_50/block3',
                image_scales=self.input_image_scales,
                iou=1.0,
                max_feature_num=self.input_max_feature_num,
                abs_thres=self.input_score_threshold,
                model_fn=model_fn)

            

            ## Optimistic restore.
            latest_checkpoint = config.model_path+'variables/variables'
            variables_to_restore = tf.global_variables()

            reader = tf.train.NewCheckpointReader(latest_checkpoint)
            saved_shapes = reader.get_variable_to_shape_map()

            variable_names_to_restore = [var.name.split(':')[0] for var in variables_to_restore]
            for shape in saved_shapes:
                if not shape in variable_names_to_restore:
                    print(shape)

            for var_name in variable_names_to_restore:
                if not var_name in saved_shapes:
                    print("WARNING. Saved weight not exists in checkpoint. Init var:", var_name)

            var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables_to_restore
                    if var.name.split(':')[0] in saved_shapes])
            restore_vars = []
            with tf.variable_scope('', reuse=True):
                for var_name, saved_var_name in var_names:
                    try:
                        curr_var = tf.get_variable(saved_var_name)
                        var_shape = curr_var.get_shape().as_list()
                        if var_shape == saved_shapes[saved_var_name]:
                            # print("restore var:", saved_var_name)
                            restore_vars.append(curr_var)
                    except ValueError:
                        print("Ignore due to ValueError on getting var:", saved_var_name) 
            saver = tf.train.Saver(restore_vars)

            sess = tf.Session()
            self.sess = sess
            # Initialize variables.
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver.restore(sess, latest_checkpoint)



           

            # # Loading model that will be used.
            # tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
            #                             config.model_path)


            graph = tf.get_default_graph()
            self.attention_score = tf.reshape(scores,
                                    [tf.shape(scores)[0]]) # remove extra dim.

            self.locations, self.descriptors = feature_extractor.DelfFeaturePostProcessing(
                boxes, features, config)
            tf.summary.scalar('input_max_feature_num', self.input_max_feature_num) # Dummy summary
            self.summary_merged = tf.summary.merge_all()
            print("self.summary_merged:", self.summary_merged)
            self.test_writer = tf.summary.FileWriter('./test_summary_log', graph)




    def get_feature_from_bytes(self, image_bytes):
        """
        loop over image_bytes, get delf feature. 
        Args:
            image_bytes: list of image bytes (JPEG)  
        """
        num_images = len(image_bytes)
        tf.logging.info('done! Found %d images', num_images)

        location_np_list = []
        descriptor_np_list = []
        feature_scale_np_list = []
        attention_score_np_list = []
        attention_np_list = []
        
        start = time.clock()
        for i in range(num_images):
            # # Get next image.
            im = self.sess.run(self.image_tf, feed_dict={self.byte_value: image_bytes[i]})
            
            # For getting more information to draw graph
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # Extract and save features.
            (summary, locations_out, descriptors_out, feature_scales_out,
                attention_score_out, attention_out) = self.sess.run(
                    [self.summary_merged, self.locations, self.descriptors, self.feature_scales, self.attention_score, self.attention],
                    feed_dict={
                        self.input_image:
                            im,
                        self.input_score_threshold:
                            self.config.delf_local_config.score_threshold,
                        self.input_image_scales:
                            list(self.config.image_scales),
                        self.input_max_feature_num:
                            self.config.delf_local_config.max_feature_num
                    },
                    options=run_options,
                    run_metadata=run_metadata)
            self.test_writer.add_run_metadata(run_metadata, 'image%d' % i)
            self.test_writer.add_summary(summary, i)
            location_np_list.append(locations_out)
            descriptor_np_list.append(descriptors_out)
            feature_scale_np_list.append(feature_scales_out)
            attention_score_np_list.append(attention_score_out)
            attention_np_list.append(attention_out)
        self.test_writer.close()

        return location_np_list, descriptor_np_list, feature_scale_np_list, attention_score_np_list, attention_np_list
        

def get_feature_from_path(image_paths, config_path):
  """
  with filename queue, batch proces. 
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  
  num_images = len(image_paths)
  tf.logging.info('done! Found %d images', num_images)

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.gfile.FastGFile(config_path, 'r') as f:
    text_format.Merge(f.read(), config)

  location_np_list = []
  descriptor_np_list = []
  feature_scale_np_list = []
  attention_np_list = []

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Reading list of images.
    filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)

    with tf.Session() as sess:
      # Initialize variables.
      init_op = tf.global_variables_initializer()
      sess.run(init_op)

      # Loading model that will be used.
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                 config.model_path)
      graph = tf.get_default_graph()
      input_image = graph.get_tensor_by_name('input_image:0')
      input_score_threshold = graph.get_tensor_by_name('input_abs_thres:0')
      input_image_scales = graph.get_tensor_by_name('input_scales:0')
      input_max_feature_num = graph.get_tensor_by_name(
          'input_max_feature_num:0')
      boxes = graph.get_tensor_by_name('boxes:0')
      raw_descriptors = graph.get_tensor_by_name('features:0')
      feature_scales = graph.get_tensor_by_name('scales:0')
      attention_with_extra_dim = graph.get_tensor_by_name('scores:0')
      attention = tf.reshape(attention_with_extra_dim,
                             [tf.shape(attention_with_extra_dim)[0]])

      locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
          boxes, raw_descriptors, config)

      # Start input enqueue threads.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      start = time.clock()
      for i in range(num_images):
        # Write to log-info once in a while.
        if i == 0:
          tf.logging.info('Starting to extract DELF features from images...')
        elif i % _STATUS_CHECK_ITERATIONS == 0:
          elapsed = (time.clock() - start)
          tf.logging.info('Processing image %d out of %d, last %d '
                          'images took %f seconds', i, num_images,
                          _STATUS_CHECK_ITERATIONS, elapsed)
          start = time.clock()

        # # Get next image.
        im = sess.run(image_tf)

        # Extract and save features.
        (locations_out, descriptors_out, feature_scales_out,
         attention_out) = sess.run(
             [locations, descriptors, feature_scales, attention],
             feed_dict={
                 input_image:
                     im,
                 input_score_threshold:
                     config.delf_local_config.score_threshold,
                 input_image_scales:
                     list(config.image_scales),
                 input_max_feature_num:
                     config.delf_local_config.max_feature_num
             })
        location_np_list.append(locations_out)
        descriptor_np_list.append(descriptors_out)
        feature_scale_np_list.append(feature_scales_out)
        attention_np_list.append(attention_out)

      # Finalize enqueue threads.
      coord.request_stop()
      coord.join(threads)

    return location_np_list, descriptor_np_list, feature_scale_np_list, attention_np_list    
    

def batch_get_feature(image_paths, config_path, output_dir, return_numpy_values=False):
  """
  with queue, batch proces. 
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  
  num_images = len(image_paths)
  tf.logging.info('done! Found %d images', num_images)

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.gfile.FastGFile(config_path, 'r') as f:
    text_format.Merge(f.read(), config)

  # Create output directory if necessary.
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)


  if return_numpy_values:
    location_np_list = []
    descriptor_np_list = []
    feature_scale_np_list = []
    attention_np_list = []

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Reading list of images.
    filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)

    with tf.Session() as sess:
      # Initialize variables.
      init_op = tf.global_variables_initializer()
      sess.run(init_op)

      # Loading model that will be used.
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                 config.model_path)
      graph = tf.get_default_graph()
      input_image = graph.get_tensor_by_name('input_image:0')
      input_score_threshold = graph.get_tensor_by_name('input_abs_thres:0')
      input_image_scales = graph.get_tensor_by_name('input_scales:0')
      input_max_feature_num = graph.get_tensor_by_name(
          'input_max_feature_num:0')
      boxes = graph.get_tensor_by_name('boxes:0')
      raw_descriptors = graph.get_tensor_by_name('features:0')
      feature_scales = graph.get_tensor_by_name('scales:0')
      attention_with_extra_dim = graph.get_tensor_by_name('scores:0')
      attention = tf.reshape(attention_with_extra_dim,
                             [tf.shape(attention_with_extra_dim)[0]])

      locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
          boxes, raw_descriptors, config)

      # Start input enqueue threads.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      start = time.clock()
      for i in range(num_images):
        # Write to log-info once in a while.
        if i == 0:
          tf.logging.info('Starting to extract DELF features from images...')
        elif i % _STATUS_CHECK_ITERATIONS == 0:
          elapsed = (time.clock() - start)
          tf.logging.info('Processing image %d out of %d, last %d '
                          'images took %f seconds', i, num_images,
                          _STATUS_CHECK_ITERATIONS, elapsed)
          start = time.clock()

        # # Get next image.
        im = sess.run(image_tf)

        # If descriptor already exists, skip its computation.
        out_desc_filename = os.path.splitext(os.path.basename(
            image_paths[i]))[0] + _DELF_EXT
        out_desc_fullpath = os.path.join(output_dir, out_desc_filename)
        if tf.gfile.Exists(out_desc_fullpath):
          tf.logging.info('Skipping %s', image_paths[i])
          continue

        # Extract and save features.
        (locations_out, descriptors_out, feature_scales_out,
         attention_out) = sess.run(
             [locations, descriptors, feature_scales, attention],
             feed_dict={
                 input_image:
                     im,
                 input_score_threshold:
                     config.delf_local_config.score_threshold,
                 input_image_scales:
                     list(config.image_scales),
                 input_max_feature_num:
                     config.delf_local_config.max_feature_num
             })
        if return_numpy_values:
          location_np_list.append(locations_out)
          descriptor_np_list.append(descriptors_out)
          feature_scale_np_list.append(feature_scales_out)
          attention_np_list.append(attention_out)

        serialized_desc = feature_io.WriteToFile(
            out_desc_fullpath, locations_out, feature_scales_out,
            descriptors_out, attention_out)

      # Finalize enqueue threads.
      coord.request_stop()
      coord.join(threads)

    if return_numpy_values:
      return location_np_list, descriptor_np_list, feature_scale_np_list, attention_np_list    
    

def main(unused_argv):


  # Read list of images.
  tf.logging.info('Reading list of images...')
  image_paths = _ReadImageList(cmd_args.list_images_path)  

  batch_get_feature(image_paths, cmd_args.config_path, cmd_args.output_dir)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--config_path',
      type=str,
      default='delf_config_example.pbtxt',
      help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
  parser.add_argument(
      '--list_images_path',
      type=str,
      default='list_images.txt',
      help="""
      Path to list of images whose DELF features will be extracted.
      """)
  parser.add_argument(
      '--output_dir',
      type=str,
      default='test_features',
      help="""
      Directory where DELF features will be written to. Each image's features
      will be written to a file with same name, and extension replaced by .delf.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)

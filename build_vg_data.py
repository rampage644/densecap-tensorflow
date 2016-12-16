# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Converts Visual Genome data to TFRecords file format with Example protos.

The raw ImageNet data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/VG_100K/713941.jpg

where '713941' is the unique image id.

Each record within the TFRecord file is a serialized Example proto.
The Example proto contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/filename: string containing the basename of the image file

  image/object/bbox/x: list of integers specifying the 1+ human annotated
    bounding boxes
  image/object/bbox/width: list of integers specifying the 1+ human annotated
    bounding boxes
  image/object/bbox/y: list of integers specifying the 1+ human annotated
    bounding boxes
  image/object/bbox/height: list of integers specifying the 1+ human annotated
    bounding boxes

Running this script using 16 threads may take around ~2.5 hours on a HP Z420.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import os
import sys
import threading
from datetime import datetime

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data/',
                           'Data directory')
tf.app.flags.DEFINE_string('output_directory', 'data/output/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_string('region_file',
                           'region_descriptions.json',
                           'Region descriptions file')


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, bbox, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    bbox: list of bounding boxes; each box is a list of integers
      specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
      the same label as the image label.
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  x = []
  y = []
  wdths = []
  hghts = []
  for b in bbox:
    assert len(b) == 4
    _ = [l.append(point) for l, point in zip([x, y, wdths, hghts], b)]

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/filename': _bytes_feature(os.path.basename(filename).encode('utf-8')),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/bbox/x': _float_feature(x),
      'image/bbox/y': _float_feature(y),
      'image/bbox/width': _float_feature(wdths),
      'image/bbox/height': _float_feature(hghts),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image



def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'r') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, data, start, end):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    humans: list of strings; each string is a human-readable label
    bboxes: list of bounding boxes for each image. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the image.
    num_shards: integer number of shards for this data set.
  """
  end = len(data) if end > len(data) else end
  output_filename = '%.5d-of-%.5d' % (start, end)
  output_file = os.path.join(FLAGS.output_directory, output_filename)
  writer = tf.python_io.TFRecordWriter(output_file)

  total = end-start
  for i in range(start, end):
    id_, regions = data[i]['id'], data[i]['regions']

    filename = '{}/{}.jpg'.format(FLAGS.data_dir, id_)
    bbox = [[r['x'], r['y'], r['width'], r['height']] for r in regions]
    image_buffer, height, width = _process_image(filename, coder)

    example = _convert_to_example(filename, image_buffer, bbox,
                                  height, width)
    writer.write(example.SerializeToString())

    # print progress for evert 10% advance
    if not ((i-start) % (total // 10)):
      print('{} [thread {}]: Wrote {} images to {}'.format(
        datetime.now(), thread_index, i-start, output_file
      ))
      sys.stdout.flush()

  writer.close()


def _process_image_files(data, num_threads):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    humans: list of strings; each string is a human-readable label
    bboxes: list of bounding boxes for each image. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the image.
    num_shards: integer number of shards for this data set.
  """
  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  step = len(data) // num_threads
  for thread_index in range(num_threads):
    start, end = thread_index * step, (thread_index + 1) * step
    args = (coder, thread_index, data, start, end)
    thread = threading.Thread(target=_process_image_files_batch, args=args)
    thread.start()
    threads.append(thread)

  # Wait for all the threads to terminate.
  coord.join(threads)
  sys.stdout.flush()


def main(_):
  '''Entry function'''
  with open(FLAGS.region_file) as ifile:
    data = json.load(ifile)
    _process_image_files(data, FLAGS.num_threads)


if __name__ == '__main__':
  tf.app.run()

import tensorflow as tf
import os
import json
from PIL import Image
import io
import numpy as np
# import progressbar

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('dataset_path','', 'Path to input dataset')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_boolean('use_negatives',False,'Whether to add images with no boxes to the dataset')
FLAGS = flags.FLAGS


def create_tf_example(example, label_map):

  # check that there are boxes, unless flag is set
  # in which case flip switch to append empty box
  empty_box = False
  if not os.path.isfile(os.path.join(example, 'data.json')):
    if FLAGS.use_negatives:
      empty_box = True
    else:
      return None

  if os.path.isfile(os.path.join(example, 'img.jpg')):
    image_format = 'jpg'
  # elif os.path.isfile(os.path.join(example, 'img.png')):
    # image_format = 'png'
  else:
    return None

  filename = os.path.join(example, 'img.{}'.format(image_format))

  with tf.gfile.GFile(filename,'rb') as fid:
      encoded_jpg = fid.read()

  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)

  if image.format != 'JPEG':
      return None

  width, height = image.size


  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  if not empty_box:
    with open(os.path.join(example, 'data.json')) as json_file:
      data = json.load(json_file)


    
    for category, boxes in data['boxes'].items():
      for box in boxes:
        # do not append invalid categories
        if label_map.has_key(category):

          # fix invalid boxes
          for key in box.keys():
            box[key] = np.clip(box[key], 0, 1)

          xmins.append(box['xmin'])
          xmaxs.append(box['xmax'])
          ymins.append(box['ymin'])
          ymaxs.append(box['ymax'])
          classes_text.append(category.encode('utf8'))
          classes.append(label_map[category])


  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):

  with open(os.path.join(FLAGS.dataset_path, 'train.txt')) as train_f:
    train_files = [x.strip() for x in train_f.readlines()]
  with open(os.path.join(FLAGS.dataset_path, 'test.txt')) as test_f:
    test_files = [x.strip() for x in test_f.readlines()]

  label_map = {}
  out_map = ''
  with open(os.path.join(FLAGS.dataset_path, 'valid_categories.txt')) as cat_f:
    for i, cat in enumerate([x.strip() for x in cat_f.readlines()]):
      label_map[cat] = i+1
      out_map += "item {\n  id: %s\n  name: '%s'\n}\n" % (i+1, cat)

  if not os.path.isdir(FLAGS.output_path):
    os.mkdir(FLAGS.output_path)

  with open(os.path.join(FLAGS.output_path, 'label_map.pbtxt'),'w') as map_f:
    map_f.write(out_map)

  train_examples = [os.path.join(FLAGS.dataset_path, 'imgs', im) for im in train_files]
  test_examples = [os.path.join(FLAGS.dataset_path, 'imgs', im) for im in test_files]

  
  train_output_path = os.path.join(FLAGS.output_path, 'train.record')
  val_output_path = os.path.join(FLAGS.output_path, 'val.record')

  # progress = progressbar.ProgressBar(widgets=[progressbar.ETA(), ' ', progressbar.Percentage()])

  train_count, val_count = 0, 0

  for path, examples in [(train_output_path, train_examples), (val_output_path, test_examples)]:
    writer = tf.python_io.TFRecordWriter(path)
    for example in examples:
      tf_example = create_tf_example(example, label_map)
      if tf_example is not None:
        if path == train_output_path:
          train_count += 1
        else:
          val_count += 1 
        writer.write(tf_example.SerializeToString())
    writer.close()
  with open(os.path.join(FLAGS.output_path, 'count.txt'),'w') as f:
    f.write("Train: {}\nVal: {}".format(train_count, val_count))

if __name__ == '__main__':
  tf.app.run()

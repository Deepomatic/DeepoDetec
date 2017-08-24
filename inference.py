import numpy as np
import os
import sys
import tensorflow as tf
import timer
import progressbar

from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("models")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

flags = tf.app.flags

flags.DEFINE_string('ckpt_path','','Path to the checkpoint to test')
flags.DEFINE_string('label_path','','Path to the labels.pbtxt file')
flags.DEFINE_string('image_path','','Path to the images to test')
flags.DEFINE_string('gpu','0','GPUs to use')
flags.DEFINE_float('gpu_mem',None,'% of GPU memory to use')

FLAGS = flags.FLAGS

def main(_):
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FLAGS.ckpt_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')


  label_map = label_map_util.load_labelmap(FLAGS.label_path)
  max_num_classes = max([item.id for item in label_map.item])
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes)
  category_index = label_map_util.create_category_index(categories)

  def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


  _timer = timer.Timer()

  with detection_graph.as_default():
    config = None if FLAGS.gpu_mem is None else tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=FLAGS.gpu_mem))
    with tf.Session(
            graph=detection_graph, 
            config=config
            ) as sess:

      progress = progressbar.ProgressBar(widgets=[progressbar.ETA(), ' ', progressbar.Percentage()])
      for image_path in progress(\
          [os.path.join(FLAGS.image_path, im) for im in os.listdir(FLAGS.image_path)]):
        

        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)


        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        _timer.tic()

        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        _timer.toc()

        print _timer.diff

        # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
            # image_np,
            # np.squeeze(boxes),
            # np.squeeze(classes).astype(np.int32),
            # np.squeeze(scores),
            # category_index,
            # use_normalized_coordinates=True,
            # line_thickness=8)
        # plt.imsave(os.path.join(FLAGS.image_path, 'detec_{}'.format(os.path.basename(image_path))), image_np)

  print "Average inference time: {}s".format(_timer.average_time)

if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
  tf.app.run()

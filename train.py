from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import os

from tfrecord_parser import parse_tfrecords
from tfrecord_creator import create_tfrecords
from utils import get_dataset

print('TensorFlow', tf.__version__)

images_path, mask_path, num_classes, dataset_size = get_dataset(dataset_path='../mini',folder='training')

batch_size = 1
H, W = 512, 512

tfrecord_dir = os.path.join(os.getcwd(), 'tfrecords')
os.makedirs(tfrecord_dir, exist_ok=True)

checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# create_tfrecords(images_path, mask_path, tfrecord_dir)


train_tfrecords = os.path.join(tfrecord_dir, 'train*.tfrecord')
input_function = parse_tfrecords(
    filenames=train_tfrecords,
    height=H,
    width=W,
    batch_size=batch_size)
print(input_function)
for data, annotation in input_function.take(1):
    image_batch = data.numpy()
    abxs_batch = annotation.numpy()
    print(data.shape)
    print(annotation)
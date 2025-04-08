TESTING= False
import sys
import tensorflow as tf
import imageio.v3 as iio
import numpy as np

def get_image():
  pass

def get_mask():
  pass

def main():
  data = iio.imread('samples/label_region_prov_Santo_Stefano_del_Sole_1_00001_0.png')
  data_labels = np.load('samples/array_mask_label_region_prov_Santo_Stefano_del_Sole_1_00001_0.npz')['arr_0']
  if TESTING:
    print(type(data))
    print(data.shape)
    #data_labels = np.load('samples/array_label_region_prov_Santo_Stefano_del_Sole_1_00001_0.npz')['arr_0']
    print(type(data_labels))
    print(data_labels.shape)
  
    tf.debugging.set_log_device_placement(True)
    print(f"TensorFlow version: {tf.__version__}")

    with tf.device('/GPU:0'):
      a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # Running on GPU
    c = tf.matmul(a, b)
    print(c)

    phys_devs = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs: {phys_devs}")
  


if __name__== "__main__":
  status = main()
  sys.exit(status)

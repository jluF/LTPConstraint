import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 2"
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)

# print(tf.test.is_gpu_available())

# 定义策略
strategy = tf.distribute.MirroredStrategy()
print("设备数量：{}".format(strategy.num_replicas_in_sync))
# from tensorflow.python.client import device_lib
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
# print(get_available_gpus())
# print(device_lib.list_local_devices())

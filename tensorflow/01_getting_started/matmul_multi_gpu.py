# Measuring performance of tf.matmul on two virtual GPUs

import tensorflow as tf
import time

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=768),
     tf.config.LogicalDeviceConfiguration(memory_limit=768)])

tf.debugging.set_log_device_placement(True)

with tf.device("/GPU:1"):
    print(f"Time measurement on GPU")
    for n in [10, 50, 100, 500, 1000]:
        a = tf.ones([n,n], tf.dtypes.float32);
        b = tf.ones([n,n], tf.dtypes.float32);
        tic = time.perf_counter()
        c = tf.matmul(a, b);
        toc = time.perf_counter()
        mem_usage = tf.config.experimental.get_memory_usage("GPU:1")
        print(f"Spent {(toc-tic)/(n*n*n):.2e} seconds per entry for multiplying {n}x{n} matrices using {mem_usage} bytes of memory")

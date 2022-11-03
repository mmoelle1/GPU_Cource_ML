# Measuring performance of tf.matmul on a virtual GPU

import tensorflow as tf
import time

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

with tf.device("/GPU:0"):
    print(f"Time measurement on GPU")
    for n in [10, 50, 100, 500, 1000, 5000, 10000]:
        a = tf.ones([n,n], tf.dtypes.float32);
        b = tf.ones([n,n], tf.dtypes.float32);
        tic = time.perf_counter()
        c = tf.matmul(a, b);
        toc = time.perf_counter()
        mem_usage = tf.config.experimental.get_memory_usage("GPU:0")
        print(f"Spent {(toc-tic)/(n*n*n):.2e} seconds per entry for multiplying {n}x{n} matrices using {mem_usage} bytes of memory")

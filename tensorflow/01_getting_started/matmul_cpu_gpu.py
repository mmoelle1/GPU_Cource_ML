# Measuring performance of tf.matmul on a CPU and a GPU

import tensorflow as tf
import time

with tf.device("/CPU:0"):
    print(f"Time measurement on CPU")
    for n in [10, 50, 100, 500, 1000, 5000, 10000]:
        a = tf.ones([n,n], tf.dtypes.float32);
        b = tf.ones([n,n], tf.dtypes.float32);
        tic = time.perf_counter()
        c = tf.matmul(a, b);
        toc = time.perf_counter()
        print(f"Spent {(toc-tic)/(n*n*n):.2e} seconds per entry for multiplying {n}x{n} matrices")

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

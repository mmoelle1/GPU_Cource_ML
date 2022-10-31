# TensorFlow and tf.keras
import tensorflow as tf

# TensorFlow datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import numpy as np
import time

from datetime import datetime

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=768),
     tf.config.LogicalDeviceConfiguration(memory_limit=768)])

# Import the Fashion MNIST dataset: 70,000 grayscale images of clothes
# (28 by 28 pixels low-resolution images) in 10 categories
(ds_train, ds_test), ds_info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Print the number of training/testing images
print(f"Using {ds_train.cardinality()} datasets for training")
print(f"Using {ds_test.cardinality()} datasets for testing")

# Specify the categories
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# The pixel values fall in the range of 0 to 255. For the classifier
# to be effective we have to rescale all values to the range of 0 to 1
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img)
ds_train = ds_train.batch(128)
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():

    # Start time measurement
    tic = time.perf_counter()
    
    # Setup the layers of the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()
    
    # Train the model
    model.fit(ds_train, epochs=10,
              callbacks = [tboard_callback])

    # Stop time measurement
    toc = time.perf_counter()
    print(f"Training completed in {(toc-tic)} seconds")
    
    # Evaluate accuracy
    test_loss, test_acc = model.evaluate(ds_test, verbose=2)
    print('\nTest accuracy:', test_acc)

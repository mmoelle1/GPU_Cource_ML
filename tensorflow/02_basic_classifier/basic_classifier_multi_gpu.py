# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=768),
     tf.config.LogicalDeviceConfiguration(memory_limit=768)])

# Import the Fashion MNIST dataset: 70,000 grayscale images of clothes
# (28 by 28 pixels low-resolution images) in 10 categories
fashion_mnist = tf.keras.datasets.fashion_mnist

# Separate the dataset into 60,000 images for training and 10,000 for testing
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Print the number of training/testing images
print(f"Using {train_images.shape} images and {len(train_labels)} labels for training")
print(f"Using {test_images.shape} images and {len(test_labels)} labels for testing")

# Specify the categories
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# The pixel values fall in the range of 0 to 255. For the classifier
# to be effective we have to rescale all values to the range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
#with tf.device("/GPU:0"):

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
    model.fit(train_images, train_labels, epochs=10)

    # Stop time measurement
    toc = time.perf_counter()
    print(f"Training completed in {(toc-tic)} seconds")
    
    # Evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    
    # Make predicitons
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    
    predictions = probability_model.predict(test_images)
    
    # Inspect prediction of first 10 test data
    for i in range(10):
        print(f"Test data #{i}")
        print(f"Confidence values for all 10 categories\n{predictions[i]}")
        print(f"Label with highest confidence value is {np.argmax(predictions[i])} correct label is {test_labels[i]}")

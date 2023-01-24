# TensorFlow
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tf_ds
import keras
from keras import layers

# Misc.
import plot_results as my_plt
import random as r

models_dir = "./models"
dataset_dir = "./datasets"
batch_size = 32

if not os.path.isdir(models_dir):
    os.mkdir(models_dir)
if not os.path.isdir(dataset_dir):
    os.mkdir(dataset_dir)


def normalize_img(image, label):
    # Normalizes images and casts image data to float32
    return tf.cast(image, tf.float32) / 255., label


def to_char(num):
    # Return the number in ASCII form
    if num < 10:
        return chr(num + 48)
    elif num < 36:
        return chr(num + 55)
    else:
        return chr(num + 61)


def main():
    # Notes:
    # - Save/Load models to streamline debugging
    # - Find out how to use Handwriting Sample Form

    # Download dataset if necessary
    tf_ds.builder("emnist").download_and_prepare(download_dir=dataset_dir)

    # Load dataset
    (train_ds, test_ds), ds_info = tf_ds.load(
        'emnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    length = len(test_ds)
    print(f"Training data size: {length}")
    print(f"Testing data size: {length}")

    # Preprocess data
    train_ds = train_ds \
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .shuffle(ds_info.splits['train'].num_examples) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds \
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    # Check if a save model exists, otherwise create a new model
    if len(os.listdir(path=models_dir)) > 0:
        print("Loading model...")
        # Load the existing model
        model = keras.models.load_model(filepath=models_dir)
        print("Model loaded!\n")
    else:
        print("Building new model...")
        # Build the network model
        model = keras.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(62, activation='softmax')
        ])

        # Compile network
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train and validate the network
        print("\nTraining model...\n")
        model.fit(
            train_ds,
            epochs=2,
            validation_data=test_ds,
            validation_batch_size=batch_size
        )

        # Save the model
        keras.models.save_model(model=model, filepath=models_dir)
        print("Model saved!\n")

    # Test the network
    print("Testing model...")
    test_results = model.evaluate(test_ds, verbose=1)
    probability_model = keras.Sequential([model, layers.Softmax()])
    predictions = probability_model.predict(test_ds, verbose=0)

    print(f"\nTesting accuracy: {(test_results[1] * 100):.2f}%")
    print("\n")

    img_arr, lbl_arr = my_plt.tfDataset2Lists(test_ds)

    while 1:
        i = r.randint(0, length)
        print("\n")
        print(f"Prediction: {to_char(int(np.argmax(predictions[i])))}")
        print(f"Actual: {to_char(lbl_arr[i])}")
        my_plt.plot_image(img_arr[i], lbl_arr[i], predictions[i])
        my_plt.plt.show()
        input("Press [ENTER] to show another plot")


if __name__ == "__main__":
    main()

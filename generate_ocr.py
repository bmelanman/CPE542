import tensorflow as tf
import tensorflow_datasets as tf_ds
from numpy import random

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input

batch_size = 16
input_size = 28

result_arr = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Numbers
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',  # Uppercase
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',  # Lowercase
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

input_shape = (input_size, input_size, 1)
output_shape = len(result_arr)


def normalize_img(image, label):
    # Normalizes images and casts image data to float32
    return tf.cast(image, tf.float32) / 255., label


def random_rotate(image):
    # Should randomly rotate an image 0, 90, 180, or 270 degrees
    return tf.image.rot90(image, random.choice([0, 1, 2, 3]))


def generate_ocr_model(filepath, epochs):

    # Creating the CNN model
    model = Sequential([
        Input(input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(output_shape, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Load training and verification data
    (train_ds, test_ds), ds_info = tf_ds.load(
        'emnist',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True,
        batch_size=batch_size
    )

    # Preprocess data
    train_ds = train_ds \
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(lambda image, label: (tf.image.transpose(image), label)) \
        .shuffle(1000) \
        .prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds \
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(lambda image, label: (tf.image.transpose(image), label)) \
        .prefetch(tf.data.AUTOTUNE)

    train_ds = train_ds.cache()
    test_ds = test_ds.cache()

    model.summary()

    print(
        "\033[A"
        "=================================================================\n"
        f"Epochs: {epochs}\n"
        "_________________________________________________________________"
    )

    input("Model ready to begin training! Press any key to begin...\n")

    # Train model
    print("Training Model...")
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
    )
    print("Done training!\n")

    # Save the model for later use
    print("Saving Model...")
    model.save(
        filepath=filepath,
    )
    print("Saved! Model generation is complete. \n")


if __name__ == "__main__":
    generate_ocr_model(filepath="./models/ocr", epochs=100)

# Iterate over the whole image (like sobel by with a 28x28 window)
# Try to get character into the center of the frame (look at averages to get centered?)
import tensorflow as tf
import tensorflow_datasets as tf_ds

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

models_dir = "./models/ocr"
batch_size = 16


def normalize_img(image, label):
    # Normalizes images and casts image data to float32
    return tf.cast(image, tf.float32) / 255., label


def ocr(filepath="./models", epochs=12):
    # Input shape could possibly be increased, could affect performance
    input_shape = (28, 28, 1)

    # Creating the CNN model
    ocr_model = Sequential([
        Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dropout(0.5),
        Dense(units=62, activation='sigmoid')
    ])

    # Compile the model
    ocr_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Load training and verification data
    (train_ds, test_ds), ds_info = tf_ds.load(
        'emnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

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

    # Train model
    ocr_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds
    )

    # Save the model for later use
    ocr_model.save(
        filepath=filepath,
    )


if __name__ == "__main__":
    ocr()

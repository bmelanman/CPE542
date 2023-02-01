import tensorflow as tf
import tensorflow_datasets as tf_ds

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

models_dir = "./models/ocr"
batch_size = 32
input_size = 64

result_arr = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Numbers
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',  # Uppercase
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',  # Lowercase
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]


def normalize_img(image, label):
    # Normalizes images and casts image data to float32
    return tf.cast(image, tf.float32) / 255., label


def ocr(filepath="./models", epochs=12):

    # Creating the CNN model
    ocr_model = Sequential([
        Conv2D(32, (3, 3), input_shape=(input_size, input_size, 1), activation='ReLU'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='ReLU'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=128, activation='ReLU'),
        Dropout(0.5),
        Dense(units=len(result_arr), activation='sigmoid')
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
        .map(lambda image, label: (tf.image.resize(image, (input_size, input_size)), label)) \
        .cache() \
        .shuffle(ds_info.splits['train'].num_examples) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds \
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(lambda image, label: (tf.image.resize(image, (input_size, input_size)), label)) \
        .cache() \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    # Train model
    ocr_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
    )

    # Save the model for later use
    ocr_model.save(
        filepath=filepath,
    )


if __name__ == "__main__":
    ocr()

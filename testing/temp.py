import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow_datasets import load


def normalize_img(image, label):
    # Normalizes images and casts image data to float32
    return tf.cast(image, tf.float32) / 255., label


def generate_ocr_model(filepath, epochs):

    # Creating the CNN model
    model = Sequential([
        Input(shape=(28, 28, 1)),
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
        Dense(62, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(
            learning_rate=0.005,
            epsilon=0.5
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Load training and verification data
    (train_ds, test_ds), ds_info = load(
        'emnist',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True,
        batch_size=16
    )

    # Preprocess data
    train_ds = train_ds \
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(lambda image, label: (tf.image.transpose(image), label), num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(1000) \
        .prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds \
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(lambda image, label: (tf.image.transpose(image), label), num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(100) \
        .prefetch(tf.data.AUTOTUNE)

    train_ds = train_ds.cache()
    test_ds = test_ds.cache()

    print(f"Number of Epochs: {epochs}")

    model.summary()

    input("Model ready to begin training! Press any key to begin...\n")

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.8,
        patience=2,
        min_lr=1e-9
    )

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=[reduce_lr]
    )

    model.save(
        filepath=filepath,
    )


if __name__ == "__main__":
    generate_ocr_model(filepath="../models/ocr", epochs=100)

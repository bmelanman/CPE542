import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow_datasets import load

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input
from keras.optimizers import Adam


def normalize_img(image, label):
    # Normalizes images and casts image data to float32
    return tf.cast(image, tf.float32) / 255., label


def generate_ocr_model(filepath, epochs):
    # Creating the CNN model
    # model = Sequential([
    #     Input(shape=(28, 28, 1)),
    #     Conv2D(32, (3, 3), activation='relu'),
    #     MaxPool2D((2, 2)),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPool2D((2, 2)),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.5),
    #     Dense(62, activation='softmax')
    # ])

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
        Dense(62, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(
            learning_rate=0.001,
            epsilon=0.1
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
        .map(lambda image, label: (tf.image.transpose(image), label)) \
        .shuffle(1000) \
        .prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds \
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(lambda image, label: (tf.image.transpose(image), label)) \
        .shuffle(100) \
        .prefetch(tf.data.AUTOTUNE)

    train_ds = train_ds.cache()
    test_ds = test_ds.cache()

    model.summary()

    input("Model ready to begin training! Press any key to begin...\n")

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.8,
        patience=2,
        min_lr=0.0001
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
    generate_ocr_model(filepath="./models/ocr", epochs=12)

# model = Sequential([
#     Input(shape=(28, 28, 1)),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPool2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPool2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPool2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(62, activation='sigmoid')
# ])
# Compiled with 'adam' and 'sparce_categorical_crossentropy'
#  1/12 - 433s 10ms/step - loss: 0.5618 - accuracy: 0.8094 - val_loss: 0.4697 - val_accuracy: 0.8343
#  2/12 - 422s 10ms/step - loss: 0.4525 - accuracy: 0.8395 - val_loss: 0.4558 - val_accuracy: 0.8382
#  3/12 - 428s 10ms/step - loss: 0.4378 - accuracy: 0.8439 - val_loss: 0.4505 - val_accuracy: 0.8406
#  4/12 - 431s 10ms/step - loss: 0.4317 - accuracy: 0.8455 - val_loss: 0.4535 - val_accuracy: 0.8408
#  5/12 - 415s 10ms/step - loss: 0.4285 - accuracy: 0.8469 - val_loss: 0.4649 - val_accuracy: 0.8380
#  6/12 - 387s  9ms/step - loss: 0.4278 - accuracy: 0.8473 - val_loss: 0.4601 - val_accuracy: 0.8410
#  7/12 - 380s  9ms/step - loss: 0.4274 - accuracy: 0.8476 - val_loss: 0.4685 - val_accuracy: 0.8391
#  8/12 - 390s  9ms/step - loss: 0.4281 - accuracy: 0.8474 - val_loss: 0.4744 - val_accuracy: 0.8375
#  9/12 - 490s 11ms/step - loss: 0.4299 - accuracy: 0.8467 - val_loss: 0.4681 - val_accuracy: 0.8397
# 10/12 - 379s  9ms/step - loss: 0.4314 - accuracy: 0.8470 - val_loss: 0.4745 - val_accuracy: 0.8381
# 11/12 - 381s  9ms/step - loss: 0.4334 - accuracy: 0.8463 - val_loss: 0.4787 - val_accuracy: 0.8350
# 12/12 - 477s 11ms/step - loss: 0.4357 - accuracy: 0.8461 - val_loss: 0.4765 - val_accuracy: 0.8374
# Final Results:
# - loss: 0.4357
# - accuracy: 0.8461

# Retest with new features:
# - optimizer = Adam(epsilon=0.1)
# - loss = 'categorical_crossentropy'
# - callbacks = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001)
#

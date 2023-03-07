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
        factor=0.5,
        patience=3,
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
    generate_ocr_model(filepath="./models/ocr", epochs=100)

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
# Epoch  1/12 - 419s 10ms/step - loss: 0.8454 - accuracy: 0.7403 - val_loss: 0.5575 - val_accuracy: 0.8079 - lr: 0.0010
# Epoch  2/12 - 380s  9ms/step - loss: 0.5155 - accuracy: 0.8210 - val_loss: 0.4906 - val_accuracy: 0.8287 - lr: 0.0010
# Epoch  3/12 - 380s  9ms/step - loss: 0.4686 - accuracy: 0.8343 - val_loss: 0.4655 - val_accuracy: 0.8356 - lr: 0.0010
# Epoch  4/12 - 376s  9ms/step - loss: 0.4440 - accuracy: 0.8414 - val_loss: 0.4519 - val_accuracy: 0.8391 - lr: 0.0010
# Epoch  5/12 - 363s  8ms/step - loss: 0.4280 - accuracy: 0.8458 - val_loss: 0.4415 - val_accuracy: 0.8419 - lr: 0.0010
# Epoch  6/12 - 365s  8ms/step - loss: 0.4163 - accuracy: 0.8493 - val_loss: 0.4337 - val_accuracy: 0.8443 - lr: 0.0010
# Epoch  7/12 - 380s  9ms/step - loss: 0.4073 - accuracy: 0.8519 - val_loss: 0.4289 - val_accuracy: 0.8457 - lr: 0.0010
# Epoch  8/12 - 373s  9ms/step - loss: 0.3998 - accuracy: 0.8542 - val_loss: 0.4245 - val_accuracy: 0.8473 - lr: 0.0010
# Epoch  9/12 - 380s  9ms/step - loss: 0.3934 - accuracy: 0.8558 - val_loss: 0.4208 - val_accuracy: 0.8478 - lr: 0.0010
# Epoch 10/12 - 379s  9ms/step - loss: 0.3880 - accuracy: 0.8572 - val_loss: 0.4188 - val_accuracy: 0.8482 - lr: 0.0010
# Epoch 11/12 - 375s  9ms/step - loss: 0.3831 - accuracy: 0.8587 - val_loss: 0.4165 - val_accuracy: 0.8492 - lr: 0.0010
# Epoch 12/12 - 369s  8ms/step - loss: 0.3787 - accuracy: 0.8600 - val_loss: 0.4162 - val_accuracy: 0.8494 - lr: 0.0010

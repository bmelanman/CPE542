import tensorflow as tf
import tensorflow_datasets as tf_ds

from tensorflow.python.framework.ops import disable_eager_execution
from keras import layers
from keras import models


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def benchmark():
    disable_eager_execution()

    (ds_train, ds_test), ds_info = tf_ds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    batch_size = 32

    ds_train = ds_train\
        .map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .cache()\
        .shuffle(ds_info.splits['train'].num_examples)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test\
        .map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)\
        .cache()\
        .prefetch(tf.data.experimental.AUTOTUNE)

    model = models.Sequential([
        # layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        # layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Dropout(0.25),
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(128, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(
        ds_train,
        epochs=2,
        validation_data=ds_test,
    )


if __name__ == "__main__":
    benchmark()

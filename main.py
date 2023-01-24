# TensorFlow
import tensorflow as tf
import tensorflow_datasets as tf_ds
import keras
from keras import layers

# Misc.
import plot_results as my_plt
import random as r


def normalize_img(image, label):
    # Normalizes images and casts image data to float32
    return tf.cast(image, tf.float32) / 255., label


def main():

    # Notes:
    # - Save/Load models to streamline debugging
    # - Find out how to use Handwriting Sample Form

    # Load dataset
    (train_ds, test_ds), ds_info = tf_ds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    batch_size = 32

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

    # Build the network model
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile network
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train and validate the network
    model.fit(
        train_ds,
        epochs=2,
        validation_data=test_ds
    )

    # Evaluate the results
    test_results = model.evaluate(test_ds, verbose=1)

    # Test the network
    probability_model = keras.Sequential([model, layers.Softmax()])
    print("\nTraining model...\n")
    predictions = probability_model.predict(test_ds, verbose=0)

    length = len(test_ds)
    print(f"Training data size: {length}")
    print(f"Testing data size: {length}")
    print(f"\nTraining accuracy: {(test_results[1] * 100):.2f}%")
    print("\n")

    while 1:
        my_plt.plot_results(r.randint(0, length), predictions, test_ds)
        input("Press [ENTER] to show another plot")


if __name__ == "__main__":
    main()
    # test()

    # length = len(test_data) - 1
    # while True:
    #     usr = input(f"Choose a value between 0 and {length}, or -1 to exit: ")
    #     if usr == '':
    #         continue
    #     i = int(usr)
    #     if i < 0 or i > length:
    #         break
    #     plot_results(i, predictions, test_data)

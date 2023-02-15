import tensorflow as tf
import tensorflow_datasets as tf_ds

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

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


def generate_ocr_model(filepath, epochs, model=None):
    if model is None:
        # Creating the CNN model
        model = Sequential([
            Conv2D(32, (3, 3), input_shape=input_shape, activation='ReLU'),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation='ReLU'),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=128, activation='ReLU'),
            Dropout(0.3),
            Dense(units=len(result_arr), activation='sigmoid')
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
        .shuffle(int(ds_info.splits['train'].num_examples / ds_info.splits['train'].num_shards)) \
        .prefetch(tf.data.AUTOTUNE) \
        .cache()

    test_ds = test_ds \
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(lambda image, label: (tf.image.transpose(image), label)) \
        .prefetch(tf.data.AUTOTUNE) \
        .cache()

    model.summary()

    input("Model ready to begin training! Press any key to begin... ")

    # Train model
    print("Training Model...")
    model.fit(
        train_ds,
        shuffle=True,
        epochs=epochs,
        validation_data=test_ds,
    )
    print("Done training!\n")

    # Save the model for later use
    print("Saving Model...")
    ocr_model.save(
        filepath=filepath,
    )
    print("Saved! Model generation is complete. \n")


if __name__ == "__main__":
    ocr_model = Sequential([
        Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        MaxPool2D(2),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPool2D(2),
        Conv2D(256, 3, padding='same', activation='relu'),
        MaxPool2D(2),
        Flatten(),
        Dense(1024, activation='elu'),
        Dropout(0.25),
        Dense(128, activation='elu'),
        Dropout(0.20),
        Dense(output_shape, activation='sigmoid')
    ])

    generate_ocr_model(filepath="../models/", epochs=12, model=ocr_model)

    # model_0 = Sequential([
    #     Conv2D(32, (3, 3), input_shape=input_shape, activation='ReLU'),
    #     MaxPool2D(pool_size=(2, 2)),
    #     Conv2D(32, (3, 3), activation='ReLU'),
    #     MaxPool2D(pool_size=(2, 2)),
    #     Flatten(),
    #     Dense(units=128, activation='ReLU'),
    #     Dropout(0.3),
    #     Dense(units=len(result_arr), activation='sigmoid')
    # ])
    # Note: This is the original design
    # Info:
    #   Epoch 1/2 - 219s 10ms/step - loss: 0.5447 - accuracy: 0.8179 - val_loss: 0.4024 - val_accuracy: 0.8549
    #   Epoch 2/2 - 187s  9ms/step - loss: 0.4184 - accuracy: 0.8504 - val_loss: 0.3850 - val_accuracy: 0.8603
    # Testing accuracy: 86.03%
    # Result: Satisfactory

    # model_1 = Sequential([
    #     MaxPool2D(pool_size=(2, 2), input_shape=(input_size, input_size, 1)),
    #     Conv2D(64, (3, 3), activation='ReLU'),
    #     MaxPool2D(pool_size=(2, 2)),
    #     Conv2D(32, (3, 3), activation='ReLU'),
    #     MaxPool2D(pool_size=(2, 2)),
    #     Flatten(),
    #     Dropout(0.2),
    #     Dense(units=128, activation='sigmoid'),
    #     Dropout(0.3),
    #     Dense(units=len(result_arr), activation='sigmoid')
    # ])
    # Info:
    #   Epoch 1/2 - 225s 10ms/step - loss: 0.6918 - accuracy: 0.7794 - val_loss: 0.4666 - val_accuracy: 0.8364
    #   Epoch 2/2 - 200s  9ms/step - loss: 0.4959 - accuracy: 0.8275 - val_loss: 0.4352 - val_accuracy: 0.8456
    # Testing accuracy: 84.56%
    # Rating: Poor, no improvement over original design

    # model_2 = Sequential([
    #     keras.Input(shape=(input_size, input_size, 1)),
    #     Conv2D(32, kernel_size=(3, 3), activation='relu'),
    #     MaxPool2D(pool_size=(2, 2)),
    #     Conv2D(64, kernel_size=(3, 3), activation='relu'),
    #     MaxPool2D(pool_size=(2, 2)),
    #     Flatten(),
    #     Dropout(0.5),
    #     Dense(units=len(result_arr), activation='softmax')
    # ])
    # Info:
    #   Epoch 1/2 - 225s 10ms/step - loss: 0.5672 - accuracy: 0.8136 - val_loss: 0.4230 - val_accuracy: 0.8521
    #   Epoch 2/2 - 199s  9ms/step - loss: 0.4416 - accuracy: 0.8454 - val_loss: 0.4051 - val_accuracy: 0.8565
    # Testing accuracy: 85.65%
    # Result: Better, improved quickly, consider further long-term testing

    # model_3 = Sequential([
    #     keras.Input(shape=input_shape),
    #     Conv2D(16, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(32, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(64, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.2),
    #     Dense(output_shape, activation='sigmoid')
    # ])
    # Info:
    #   Epoch 1/2 - 229s 10ms/step - loss: 0.5125 - accuracy: 0.8253 - val_loss: 0.3987 - val_accuracy: 0.8567
    #   Epoch 2/2 - 197s  9ms/step - loss: 0.4023 - accuracy: 0.8546 - val_loss: 0.3741 - val_accuracy: 0.8630
    # Testing accuracy: 86.30%
    # Result: Finally, improvement over the original design!

    # model_4 = Sequential([
    #     keras.Input(shape=input_shape),
    #     Conv2D(16, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(32, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(64, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.3),
    #     Dense(output_shape, activation='softmax')
    # ])
    # Note: Small modification of prev. design: increased dropout and changed output layer activation
    # Info:
    #   Epoch 1/2 - 240s 11ms/step - loss: 0.5350 - accuracy: 0.8200 - val_loss: 0.4052 - val_accuracy: 0.8546
    #   Epoch 2/2 - 207s  9ms/step - loss: 0.4126 - accuracy: 0.8521 - val_loss: 0.3869 - val_accuracy: 0.8598
    # Testing accuracy: 85.98%
    # Result: Regression of performance, failure

    # model_5 = Sequential([
    #     keras.Input(shape=input_shape),
    #     Conv2D(16, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(32, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(64, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.3),
    #     Dense(output_shape, activation='sigmoid')
    # ])
    # Note: Changed output layer activation from softmax back to sigmoid to observe activation effects
    # Info:
    #   Epoch 1/2 - 242s 11ms/step - loss: 0.5277 - accuracy: 0.8214 - val_loss: 0.3941 - val_accuracy: 0.8561
    #   Epoch 2/2 - 206s  9ms/step - loss: 0.4094 - accuracy: 0.8530 - val_loss: 0.3791 - val_accuracy: 0.8615
    # Testing accuracy: 86.15%
    # Result: Sigmoid seems to have a notable effect on performance

    # model_6 = Sequential([
    #     keras.Input(shape=input_shape),
    #     Conv2D(16, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(32, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(64, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(128, 3, padding='same', activation='relu'),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.3),
    #     Dense(128, activation='relu'),
    #     Dense(output_shape, activation='sigmoid')
    # ])
    # Info: Added another Conv2D layer but without a MaxPool2D layer, added another Dense layer after Dropout layer
    #   Epoch 1/2 - 279s 13ms/step - loss: 0.5268 - accuracy: 0.8216 - val_loss: 0.3961 - val_accuracy: 0.8572
    #   Epoch 2/2 - 238s 11ms/step - loss: 0.4055 - accuracy: 0.8547 - val_loss: 0.3870 - val_accuracy: 0.8600
    # Testing accuracy: 86.00%
    # Result: Better, but not the best so far

    # model_7 = Sequential([
    #     keras.Input(shape=input_shape),
    #     Conv2D(16, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(32, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(64, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(128, 3, padding='same', activation='relu'),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.2),
    #     Dense(128, activation='relu'),
    #     Dropout(0.2),
    #     Dense(output_shape, activation='sigmoid')
    # ])
    # Info: Reduced dropout and added another Dropout layer after second dense layer
    #   Epoch 1/2 -
    #   Epoch 2/2 -
    # Testing accuracy:
    # Result:

    # model_8 = Sequential([
    #     keras.Input(shape=input_shape),
    #     Conv2D(16, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(32, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(64, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(128, 3, padding='same', activation='relu'),
    #     Flatten(),
    #     Dense(128, activation='elu'),
    #     Dropout(0.2),
    #     Dense(128, activation='elu'),
    #     Dropout(0.2),
    #     Dense(128, activation='elu'),
    #     Dense(output_shape, activation='sigmoid')
    # ])
    # Info: Modified Dense layer activation types
    #   Epoch 1/2 - 408s 19ms/step - loss: 0.4979 - accuracy: 0.8276 - val_loss: 0.4230 - val_accuracy: 0.8466
    #   Epoch 2/2 - 378s 17ms/step - loss: 0.4085 - accuracy: 0.8525 - val_loss: 0.4038 - val_accuracy: 0.8534
    # Testing accuracy: 85.34%
    # Result: Step time is longer, accuracy seems in a good range

    # model_9 = Sequential([
    #     keras.Input(shape=input_shape),
    #     Conv2D(16, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(32, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Dropout(0.3),
    #     Conv2D(64, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(128, 3, padding='same', activation='relu'),
    #     Flatten(),
    #     Dense(128, activation='elu'),
    #     Dense(output_shape, activation='sigmoid')
    # ])
    # Info: Added dropout layer after pool #2 (recommended by "the internet")
    #   Epoch 1/2 - 504s 11ms/step - loss: 0.4875 - accuracy: 0.8293 - val_loss: 0.4151 - val_accuracy: 0.8460
    #   Epoch 2/2 - 500s 11ms/step - loss: 0.4140 - accuracy: 0.8491 - val_loss: 0.4082 - val_accuracy: 0.8499
    # Testing accuracy:
    # Result:

    # model_a = Sequential([
    #     keras.Input(shape=input_shape),
    #     Conv2D(16, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(32, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(64, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Flatten(),
    #     Dense(128, activation=keras.layers.PReLU()),
    #     Dropout(0.2),
    #     Dense(output_shape, activation='sigmoid')
    # ])
    # Info: Using PReLu activation on Dense layer
    #   Epoch 1/2 - 490s 11ms/step - loss: 0.4845 - accuracy: 0.8321 - val_loss: 0.3948 - val_accuracy: 0.8574
    #   Epoch 2/2 - 471s 11ms/step - loss: 0.3954 - accuracy: 0.8564 - val_loss: 0.3867 - val_accuracy: 0.8592
    # Testing accuracy: 85.92%
    # Result: Great!

    # model_b = Sequential([
    #     keras.Input(shape=input_shape),
    #     Conv2D(16, 3, padding='same', activation=keras.layers.PReLU()),
    #     MaxPool2D(2),
    #     Conv2D(32, 3, padding='same', activation=keras.layers.PReLU()),
    #     MaxPool2D(2),
    #     Conv2D(64, 3, padding='same', activation=keras.layers.PReLU()),
    #     MaxPool2D(2),
    #     Flatten(),
    #     Dense(128, activation=keras.layers.PReLU()),
    #     Dropout(0.2),
    #     Dense(output_shape, activation='sigmoid')
    # ])
    # Info: Testing further use of PReLu in all layers
    #   Epoch 1/2 - 594s 13ms/step - loss: 0.4836 - accuracy: 0.8331 - val_loss: 0.4037 - val_accuracy: 0.8559
    #   Epoch 2/2 - 566s 13ms/step - loss: 0.4011 - accuracy: 0.8549 - val_loss: 0.3962 - val_accuracy: 0.8572
    # Testing accuracy: 85.72%
    # Result: Did not see notable performance increase, further testing will be done on prev. model

    # model_c = Sequential([
    #     Input(shape=input_shape),
    #     Conv2D(16, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(32, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Conv2D(64, 3, padding='same', activation='relu'),
    #     MaxPool2D(2),
    #     Flatten(),
    #     Dense(128, activation=PReLU()),
    #     Dropout(0.2),
    #     Dense(output_shape, activation='sigmoid')
    # ])
    # Info: Testing model_a with epochs raised to 6
    #   Epoch 1/6 - 496s 11ms/step - loss: 0.4806 - accuracy: 0.8330 - val_loss: 0.3928 - val_accuracy: 0.8560
    #   Epoch 2/6 - 466s 11ms/step - loss: 0.3950 - accuracy: 0.8560 - val_loss: 0.3832 - val_accuracy: 0.8584
    #   Epoch 3/6 - 468s 11ms/step - loss: 0.3798 - accuracy: 0.8606 - val_loss: 0.3828 - val_accuracy: 0.8594
    #   Epoch 4/6 - 475s 11ms/step - loss: 0.3732 - accuracy: 0.8624 - val_loss: 0.3863 - val_accuracy: 0.8595
    #   Epoch 5/6 - 475s 11ms/step - loss: 0.3699 - accuracy: 0.8636 - val_loss: 0.3867 - val_accuracy: 0.8595
    #   Epoch 6/6 -
    # Testing accuracy:
    # Result:

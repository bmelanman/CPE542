import tensorflow_datasets as tf_ds
import matplotlib.pyplot as plt
import numpy as np

from main import to_char


def tfDataset2Lists(dataset):
    # Converts tf dataset to individual lists of images and corresponding labels
    lst = list(map(list, zip(*list(tf_ds.as_numpy(dataset)))))
    return np.concatenate(lst[0], axis=0), np.concatenate(lst[1], axis=0)


def plot_image(image, image_label, prediction):
    # Prep plot
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # Plot given image
    plt.imshow(np.transpose(image, axes=[1, 0, 2]), cmap=plt.cm.binary)

    # Label image and use color to indicate accuracy
    pred_label = np.argmax(prediction)
    if pred_label == image_label:
        c = 'blue'
    else:
        c = 'red'

    plt.xlabel(
        f"Prediction: {to_char(pred_label)} ({(100 * np.max(prediction)):2.0f}%)\n"
        f"Actual: {to_char(image_label)}",
        color=c
    )


def plot_confidence(prediction, true_label):
    # Get answer range (0-9, A-Z, etc...)
    range_len = len(prediction)

    # Set up the bar graph
    plt.grid(False)
    plt.yticks([x / 10 for x in list(range(11))], [(str(x * 10) + '%') for x in list(range(11))])
    plt.xticks(range(range_len))
    plt.ylim([0, 1])

    # Plot bar graph
    this_plot = plt.bar(range(range_len), prediction, color="#777777")

    # Get the most confident prediction
    predicted_label = np.argmax(prediction)

    # Label the predicted value, then the real value,
    # and if the values are the same, it will show just blue.
    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')


def plot_results(i, prediction_arr, dataset):

    # Convert the dataset to two arrays
    img_arr, lbl_arr = tfDataset2Lists(dataset)

    # Create a new plot
    plt.figure(figsize=(6, 3))

    # Plot the given image
    plt.subplot(1, 2, 1)
    plot_image(img_arr[i], lbl_arr[i], prediction_arr[i])
    plt.title(f"Figure #{plt.gcf().number}")

    # Plot the confidence of the prediction
    plt.subplot(1, 2, 2)
    plot_confidence(prediction_arr[i], lbl_arr[i])
    plt.title(f"Test #{i}")

    plt.show()

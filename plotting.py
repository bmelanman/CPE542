import numpy as np
import matplotlib.pyplot as plt


def plot_image(predictions_array, img, true_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(
        f"{predicted_label} {(100 * np.max(predictions_array)):2.0f}% ({true_label})",
        color=color
    )


def plot_value_array(predictions_array, true_label):
    range_len = len(predictions_array)
    plt.grid(False)
    plt.xticks(range(range_len))
    plt.yticks([])
    this_plot = plt.bar(range(range_len), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')


def plot_results(i, predictions, data_list):
    image = data_list[i][0][0]
    label = data_list[i][1][0]
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plot_image(predictions[i], image, label)

    plt.subplot(1, 2, 2)
    plot_value_array(predictions[i], label)
    plt.show()

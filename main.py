import os.path
import keras
import numpy as np

from generateModel_OCR import ocr
from PIL import Image
from skimage import transform

model_dir = "./models/ocr"
data_dir = "./datasets"
test_img_dir = "./test_images/ocr_test.png"
batch_size = 16
result_arr = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Numbers
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',  # Uppercase
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',  # Lowercase
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
model_arr = []


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (28, 28, 1))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def main(new_model=False):
    if not os.path.isdir(model_dir):
        print(f"Error! Please create the directory '{model_dir}'")

    if len(os.listdir(path=model_dir)) == 0 or new_model is True:
        print("Building new model...")
        ocr(filepath=model_dir, epochs=2)
        print("Model Built!\n")

    print("Loading model...")
    ocr_model = keras.models.load_model(filepath=model_dir)
    print("Model loaded!\n")

    # Test the network
    print("Testing model...")
    # test_image = load_img(
    #     test_img_dir,
    #     color_mode='grayscale',
    #     target_size=(28, 28)
    # )
    test_image = load(test_img_dir)
    prediction = ocr_model.predict(test_image)
    np.reshape(prediction, 36)

    if np.amax(test_image) != np.amin(test_image):
        index = np.where(prediction == np.amax(prediction))
        model_arr.append(result_arr[index[1][0]])

    print(model_arr)

    # image = load('my_file.jpg')
    # model.predict(image)


if __name__ == "__main__":
    main(new_model=True)

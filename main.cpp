//
// Created by Bryce Melander on 2/13/23.
//

#include "main.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

using namespace cv;

int main() {

    printf("%s\n", std::__fs::filesystem::current_path().c_str());

    Mat img = imread("../test_images/performance.png");

    if (img.empty()) {
        printf("imgread failure!\n");
        exit(EXIT_FAILURE);
    }

    String windowName = "Performance"; //Name of the window

    namedWindow(windowName); // Create a window

    imshow(windowName, img); // Show our image inside the created window.

    waitKey(0); // Wait for any keystroke in the window

    destroyAllWindows();

    return 0;

}
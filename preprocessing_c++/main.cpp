#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat preprocess_image_predict(const std::string &image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Failed to open the image file at " << image_path << std::endl;
        exit(1);
    }
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(500, 308));
    cv::Mat gray_image;
    cv::cvtColor(resized_image, gray_image, cv::COLOR_BGR2GRAY);
    cv::Mat adjusted_image;
    cv::convertScaleAbs(gray_image, adjusted_image, 2.2, 1);
    cv::Mat denoised_image;
    cv::fastNlMeansDenoising(adjusted_image, denoised_image, 1, 7, 1);
    cv::Mat binary_image;
    cv::threshold(denoised_image, binary_image, 210, 255, cv::THRESH_BINARY);
    return binary_image;
}




int main() {
    // replace it with your image path
    std::string image_path = "/media/khadija/3E2CC5A32CC55715/GitHub/OCR/preprocessing/id.png"; 
    cv::Mat preprocessed_image = preprocess_image_predict(image_path);


    // Display the preprocessed image
    cv::namedWindow("Preprocessed Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Preprocessed Image", preprocessed_image);
    cv::waitKey(0);

    // Save the preprocessed image
    cv::imwrite("preprocessed_image.jpg", preprocessed_image);

    return 0;
}
cmake_minimum_required(VERSION 3.0)
project(PreprocessImage)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the documentation for the FindOpenCV module.
find_package(OpenCV REQUIRED)

# Declare the executable target built from your sources
add_executable(PreprocessImage main.cpp)

# Link your application with OpenCV libraries
target_link_libraries(PreprocessImage ${OpenCV_LIBS})
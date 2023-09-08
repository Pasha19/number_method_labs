cmake_minimum_required(VERSION 3.24)
project(number_methods_labs LANGUAGES CXX)
include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest
    GIT_TAG "v2.4.11"
)
FetchContent_MakeAvailable(doctest)
include_directories(.)
add_subdirectory(lab1)

cmake_minimum_required(VERSION 3.20)
project(qr_book LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(qr_book STATIC src/book.cpp)
target_include_directories(qr_book PUBLIC src)

add_executable(book_replay src/main.cpp)
target_link_libraries(book_replay PRIVATE qr_book)

enable_testing()
add_executable(test_book tests/test_book.cpp)
target_link_libraries(test_book PRIVATE qr_book)
add_test(NAME book_basic COMMAND test_book)

cmake_minimum_required(VERSION 3.14)
project(knn_kdtree_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CGAL REQUIRED)

add_executable(knn_kdtree_test main_regression.cpp)
target_link_libraries(knn_kdtree_test CGAL::CGAL)
cmake_minimum_required(VERSION 3.14)

find_package(CGAL REQUIRED)

add_executable(kdtree_viz kdtree_viz.cpp)
target_link_libraries(kdtree_viz CGAL::CGAL)

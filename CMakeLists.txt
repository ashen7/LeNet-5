#项目名称 
project(DNN)
#cmake的最低版本
cmake_minimum_required(VERSION 3.5.1)

#添加选项 支持c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -g -O3 -W -Wall")

#Release mode
set(CMAKE_BUILD_TYPE Release) 
set(current_dir ./)
set(current_include include)
set(third_party_library /home/yipeng/thirdlib)
set(lenet_source train_and_evaluate_lenet5.cpp 
                 src/full_connected_layer.cpp 
                 src/neural_network.cpp
                 src/lenet.cpp
                 src/filter.cpp
                 src/max_pooling_layer.cpp
                 src/convolutional_layer.cpp
                 tools/read_mnist.cpp)
set(link_library glog gflags pthread)

#自动查找构建工程所需程序库 找到后会预定义几个变量
find_package(OpenCV REQUIRED)

#头文件的路径 这里用了相对路径
include_directories(${current_include}
                    ${current_dir}
                    ${OpenCV_INCLUDE_DIRS}
                    ${third_party_library}/glog/include
                    ${third_party_library}/gflags/include)

#库文件的路径
link_directories(${OpenCV_LIBRARY_DIRS}
                 ${third_party_library}/glog/lib
                 ${third_party_library}/gflags/lib)

#可执行文件的名称
add_executable(lenet ${lenet_source})
target_link_libraries(lenet ${link_library} ${OpenCV_LIBS})






	


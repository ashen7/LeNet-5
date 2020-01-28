GPU=1
CUDNN=1
OPENCV=1
OPENMP=1
DEBUG=0
third_party_library=/home/yipeng/thirdlib

#1. gcc 编译host端 debug就是多加一个内存泄漏的check
if [ -n $DEBUG -a $DEBUG = 1 ]; then
    g++ -fsanitize=address -fno-omit-frame-pointer -DGPU=$GPU -DCUDNN=$CUDNN -DOPENCV=$OPENCV -DOPENMP=$OPENMPDDEBUG  \
    -c -g -O3 -W -Wall -Wfatal-errors -fopenmp -std=gnu++11 train_and_evaluate_lenet5.cpp \
    src/lenet.cpp src/convolutional_layer.cpp src/filter.cpp src/max_pooling_layer.cpp \
    src/neural_network.cpp src/full_connected_layer.cpp tools/read_mnist.cpp \
    -I ./ -I ./include -I $third_party_library/glog/include/ -I $third_party_library/gflags/include/ \
    -I $third_party_library/opencv/include/ 
else
    g++ -DGPU=$GPU -DCUDNN=$CUDNN -DOPENCV=$OPENCV -DOPENMP=$OPENMP -c -g -O3 -w -fopenmp -std=gnu++11 train_and_evaluate_lenet5.cpp \
    src/lenet.cpp src/convolutional_layer.cpp src/filter.cpp src/max_pooling_layer.cpp \
    src/neural_network.cpp src/full_connected_layer.cpp tools/read_mnist.cpp \
    -I ./ -I ./include -I $third_party_library/glog/include/ -I $third_party_library/gflags/include/ \
    -I $third_party_library/opencv/include/ 
fi

#2. nvcc 编译device端
if [ -n $DEBUG -a $DEBUG = 1 ]; then
    nvcc -fsanitize=address -fno-omit-frame-pointer --compiler-options "-Wall -Wfatal-errors -Ofast" -c -g -O3 utility/matrix_gpu.cu -I ./ -I $third_party_library/glog/include 
else
    nvcc -c -O3 --compiler-options "-Wall -Wfatal-errors -Ofast" utility/matrix_gpu.cu -I ./ -I $third_party_library/glog/include 
fi

#3. gcc link所有目标文件和库 生成可执行文件lenet5
if [ -n $DEBUG -a $DEBUG = 1 ]; then
    echo "current mode is debug"
    g++ -fsanitize=address -fopenmp -o lenet5 train_and_evaluate_lenet5.o lenet.o convolutional_layer.o filter.o max_pooling_layer.o \
    neural_network.o full_connected_layer.o read_mnist.o matrix_gpu.o -L $third_party_library/glog/lib/ \
    -L $third_party_library/gflags/lib/ -L $third_party_library/opencv/lib/ -L /usr/local/cuda/lib64 \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect \
    -lopencv_videoio -lglog -lgflags -lcudart -lpthread \
    && ./lenet5 -flagfile=conf/lenet5_flagfile_configure && rm lenet5
else
    echo "current mode is release"
    g++ -fopenmp -o lenet5 train_and_evaluate_lenet5.o lenet.o convolutional_layer.o filter.o max_pooling_layer.o \
    neural_network.o full_connected_layer.o read_mnist.o matrix_gpu.o -L $third_party_library/glog/lib/ \
    -L $third_party_library/gflags/lib/ -L $third_party_library/opencv/lib/ -L /usr/local/cuda/lib64 \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect \
    -lopencv_videoio -lglog -lgflags -lcudart -lpthread \
    && ./lenet5 -flagfile=conf/lenet5_flagfile_configure && rm lenet5
fi


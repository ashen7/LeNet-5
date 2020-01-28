GPU=1
CUDNN=1
OPENCV=1
OPENMP=1
DEBUG=0
third_party_library=/home/yipeng/thirdlib

#1. gcc 编译host端 debug就是多加一个内存泄漏的check
if [ -n $DEBUG -a $DEBUG = 1 ]; then
    g++ -fPIC -O3 -fsanitize=address -fno-omit-frame-pointer -DGPU=$GPU -DCUDNN=$CUDNN -DOPENCV=$OPENCV -DOPENMP=$OPENMPDDEBUG -c -g -O3 -W -Wall -fopenmp -std=gnu++11  \
    src/lenet.cpp src/convolutional_layer.cpp src/filter.cpp src/max_pooling_layer.cpp \
    src/neural_network.cpp src/full_connected_layer.cpp tools/read_mnist.cpp \
    -I ./ -I ./include -I $third_party_library/glog/include/ -I $third_party_library/gflags/include/ \
    -I $third_party_library/opencv/include/ 
else
    g++ -fPIC -DGPU=$GPU -DCUDNN=$CUDNN -DOPENCV=$OPENCV -DOPENMP=$OPENMP -c -O3 -W -Wall -fopenmp -std=gnu++11 \
    src/lenet.cpp src/convolutional_layer.cpp src/filter.cpp src/max_pooling_layer.cpp \
    src/neural_network.cpp src/full_connected_layer.cpp tools/read_mnist.cpp \
    -I ./ -I ./include -I $third_party_library/glog/include/ -I $third_party_library/gflags/include/ \
    -I $third_party_library/opencv/include/ 
fi

#2. nvcc 编译device端
if [ -n $DEBUG -a $DEBUG = 1 ]; then
    nvcc -fsanitize=address -fno-omit-frame-pointer -c -g -O3 --compiler-options "-Wall -Wfatal-errors -Ofast -fPIC" utility/matrix_gpu.cu -I ./ -I $third_party_library/glog/include 
else
    nvcc -c -O3 --compiler-options "-Wall -Wfatal-errors -Ofast -fPIC" utility/matrix_gpu.cu -I ./ -I $third_party_library/glog/include 
fi

#3. gcc link所有目标文件和库 生成动态链接库liblenet5.so
if [ -n $DEBUG -a $DEBUG = 1 ]; then
    g++ -shared -fsanitize=address -fopenmp -o liblenet5.so lenet.o convolutional_layer.o filter.o max_pooling_layer.o \
    neural_network.o full_connected_layer.o read_mnist.o matrix_gpu.o -L $third_party_library/glog/lib/ \
    -L $third_party_library/gflags/lib/ -L $third_party_library/opencv/lib/ -L /usr/local/cuda/lib64 \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect \
    -lopencv_videoio -lglog -lgflags -lcudart -lpthread 
else
    g++ -shared -fopenmp -o liblenet5.so lenet.o convolutional_layer.o filter.o max_pooling_layer.o \
    neural_network.o full_connected_layer.o read_mnist.o matrix_gpu.o -L $third_party_library/glog/lib/ \
    -L $third_party_library/gflags/lib/ -L $third_party_library/opencv/lib/ -L /usr/local/cuda/lib64 \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect \
    -lopencv_videoio -lglog -lgflags -lcudart -lpthread 
fi

#4. gcc 编译源文件 link动态链接库 生成可执行文件
if [ -n $DEBUG -a $DEBUG = 1 ]; then
    echo "current mode is debug"
    g++ -o lenet5 train_and_evaluate_lenet5.cpp -O3 -fsanitize=address -fno-omit-frame-pointer -DGPU=$GPU -DCUDNN=$CUDNN -DOPENCV=$OPENCV \
    -DOPENMP=$OPENMPDDEBUG -g -O3 -W -Wall -fopenmp -std=gnu++11  \
    -I ./ -I ./include -I $third_party_library/glog/include/ -I $third_party_library/gflags/include/ \
    -I $third_party_library/opencv/include/ -L ./ -L $third_party_library/glog/lib/ \
    -L $third_party_library/gflags/lib/ -L $third_party_library/opencv/lib/ -L /usr/local/cuda/lib64 \
    -llenet5 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect \
    -lopencv_videoio -lglog -lgflags -lcudart -lpthread \
    && ./lenet5 -flagfile=conf/lenet5_flagfile_configure && rm lenet5
else
    echo "current mode is release"
    g++ -o lenet5 train_and_evaluate_lenet5.cpp -O3  -DGPU=$GPU -DCUDNN=$CUDNN -DOPENCV=$OPENCV \
    -DOPENMP=$OPENMPDDEBUG  -O3 -W -Wall -fopenmp -std=gnu++11  \
    -I ./ -I ./include -I $third_party_library/glog/include/ -I $third_party_library/gflags/include/ \
    -I $third_party_library/opencv/include/ -L ./ -L $third_party_library/glog/lib/ \
    -L $third_party_library/gflags/lib/ -L $third_party_library/opencv/lib/ -L /usr/local/cuda/lib64 \
    -llenet5 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect \
    -lopencv_videoio -lglog -lgflags -lcudart -lpthread   \
    && ./lenet5 -flagfile=conf/lenet5_flagfile_configure && rm lenet5
fi

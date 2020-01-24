third_party_library=/home/yipeng/thirdlib

g++ -g -O3 -w -fopenmp test_lenet5.cpp src/lenet.cpp src/convolutional_layer.cpp src/filter.cpp \
src/max_pooling_layer.cpp src/neural_network.cpp src/full_connected_layer.cpp tools/read_mnist.cpp -o lenet5_test \
-I ./ -I ./include -I $third_party_library/glog/include/ -I $third_party_library/gflags/include/ \
-I $third_party_library/opencv/include/ -L $third_party_library/glog/lib/ \
-L $third_party_library/gflags/lib/ -L $third_party_library/opencv/lib/ \
-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect \
-lopencv_videoio -lglog -lgflags -std=gnu++11 -lpthread \
&& ./lenet5_test -flagfile=conf/lenet5_flagfile_configure && rm lenet5_test

/*
 * =====================================================================================
 *
 *       Filename:  test_lenet5.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2019年1月3日 20时32分52秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <time.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <signal.h>

#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <chrono>
#include <ratio>
#include <thread>
#include <atomic>

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <omp.h>

#include "read_mnist.h"
#include "lenet.h"
#include "convolutional_layer.h"
#include "max_pooling_layer.h"
#include "neural_network.h"
#include "full_connected_layer.h"
#include "utility/matrix_math_function.hpp"

//Google GFlags
DEFINE_int32(conv1_input_height,  28, "LeNet-5 Conv1 input height");
DEFINE_int32(conv1_input_width,    1, "LeNet-5 Conv1 input width");
DEFINE_int32(conv1_channel_number, 1, "LeNet-5 Conv1 input channel number");
DEFINE_int32(conv1_filter_height,  5, "LeNet-5 Conv1 filter height");
DEFINE_int32(conv1_filter_width,   5, "LeNet-5 Conv1 filter width");
DEFINE_int32(conv1_filter_number, 32, "LeNet-5 Conv1 filter number");
DEFINE_int32(conv1_zero_padding,   2, "LeNet-5 Conv1 zero padding");
DEFINE_int32(conv1_stride,         1, "LeNet-5 Conv1 stride");

DEFINE_int32(pool2_filter_height,  2, "LeNet-5 Pool2 filter height");
DEFINE_int32(pool2_filter_width,   2, "LeNet-5 Pool2 filter width");
DEFINE_int32(pool2_stride,         2, "LeNet-5 Pool2 stride");

DEFINE_int32(conv3_filter_height,  5, "LeNet-5 Conv3 filter height");
DEFINE_int32(conv3_filter_width,   5, "LeNet-5 Conv3 filter width");
DEFINE_int32(conv3_filter_number, 64, "LeNet-5 Conv3 filter number");
DEFINE_int32(conv3_zero_padding,   0, "LeNet-5 Conv3 zero padding");
DEFINE_int32(conv3_stride,         1, "LeNet-5 Conv3 stride");

DEFINE_int32(pool4_filter_height,  2, "LeNet-5 Pool4 filter height");
DEFINE_int32(pool4_filter_width,   2, "LeNet-5 Pool4 filter width");
DEFINE_int32(pool4_stride,         2, "LeNet-5 Pool4 stride");

DEFINE_int32(fc5_output_node,    512, "LeNet-5 fc5 output node");
DEFINE_int32(fc6_output_node,     10, "LeNet-5 fc6 output node");

DEFINE_int32(batch_size,         100, "LeNet-5 model train batch size");
DEFINE_int32(epoch,                0, "LeNet-5 model train iterator epoch");
DEFINE_double(learning_rate,   0.001, "LeNet-5 model train learning rate");

DEFINE_int32(mnist_training_data_size, 10000, "Mnist Data Set Training Picture Number");
DEFINE_int32(mnist_test_data_size,      1000, "Mnist Data Set Test Picture Number");

DEFINE_string(lenet_weights_output_file, "./backup/lenet5.weights", "LeNet-5 model train save weights file");

static int Test(std::string mnist_test_path);

void SignalAction(int signum, siginfo_t* siginfo, void* ucontext) {
    switch (signum) {
        case SIGINT:
        case SIGTERM:
        case SIGUSR1:
        case SIGUSR2:
            LOG(INFO) << "successfully capture signal";
            SingletonLeNet::Instance().set_stop_flag(true);
            break;
        default:
            break;
    }
}

//测试模型   用错误率来对网络进行评估
int Test(std::string mnist_test_path) {
    //得到测试数据集
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> mnist_test_sample_data_set; 
    std::vector<std::vector<std::vector<double>>> mnist_test_label_data_set;
    if (-1 == GetMnistTestDataSet(mnist_test_path, FLAGS_mnist_test_data_size, 
                                  mnist_test_sample_data_set, 
                                  mnist_test_label_data_set)) {
        LOG(ERROR) << "get mnist traing data set failed...";
        return -1;
    }

    double error = 0.0;
    size_t test_data_set_total = mnist_test_sample_data_set.size();
    int label = -1;
    int output = -1;
    std::vector<std::vector<double>> output_array;

    //初始化LeNet-5
    if (-1 == SingletonLeNet::Instance().Initialize(FLAGS_conv1_input_height,  FLAGS_conv1_input_width, 
                                                    FLAGS_conv1_channel_number,FLAGS_conv1_filter_height,
                                                    FLAGS_conv1_filter_width,  FLAGS_conv1_filter_number, 
                                                    FLAGS_conv1_zero_padding, FLAGS_conv1_stride, 
                                                    FLAGS_pool2_filter_height, FLAGS_pool2_filter_width,
                                                    FLAGS_pool2_stride,        
                                                    FLAGS_conv3_filter_height, FLAGS_conv3_filter_width,
                                                    FLAGS_conv3_filter_number, FLAGS_conv3_zero_padding,
                                                    FLAGS_conv3_stride, 
                                                    FLAGS_pool4_filter_height, FLAGS_pool4_filter_width, 
                                                    FLAGS_pool4_stride, 
                                                    FLAGS_fc5_output_node,
                                                    FLAGS_fc6_output_node, 
                                                    mnist_test_sample_data_set[0])) {
        LOG(ERROR) << "LeNet-5 initializer failed, programm is exiting";
        return -1;
    }
    
    //遍历测试数据集 做预测 查看结果 
    for (int i = 0; i < test_data_set_total; i++) {
        label = GetPredictResult(mnist_test_label_data_set[i]);
        SingletonLeNet::Instance().Predict(mnist_test_sample_data_set[i], 
                                           output_array);
        output = GetPredictResult(output_array);
        if (label != output) {
            error += 1.0;
        }
    }
    
    double error_ratio = error / test_data_set_total;
    LOG(WARNING) << "current test error ratio is: %" << error_ratio * 100;

    return 0;
}

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "google_logging/");
    FLAGS_stderrthreshold = 0;
    FLAGS_colorlogtostderr = true;
    
    //记录开始时间
    std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();

    std::string mnist_path = "./data/";
    std::string mnist_test_path = mnist_path + "mnist_test_set/";
    Test(mnist_test_path);
    
    //记录结束时间
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    //设置单位为秒
    std::chrono::duration<int, std::ratio<1, 1>> sec = std::chrono::duration_cast<
                                                       std::chrono::seconds>(end - begin);
    //打印耗时
    LOG(INFO) << "programming is exiting, the total time is: " << sec.count() << "s";
    google::ShutdownGoogleLogging();
    
    return 0;

}


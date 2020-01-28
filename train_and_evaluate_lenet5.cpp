/*
 * =====================================================================================
 *
 *       Filename:  train_and_evaluate_lenet5.cpp
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
#include "utility/matrix_gpu.h"
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

DEFINE_bool(stop_flag, false, "programming stop flag");
DEFINE_string(lenet_weights_output_file, "./backup/lenet5.weights", "LeNet-5 model train save weights file");

void SignalAction(int signum, siginfo_t* siginfo, void* ucontext) {
    switch (signum) {
        case SIGINT:
        case SIGTERM:
        case SIGUSR1:
        case SIGUSR2:
            LOG(INFO) << "successfully capture signal";
            SingletonLeNet::Instance().set_stop_flag(true);
            FLAGS_stop_flag = true;
            break;
        default:
            break;
    }
}

//训练和评估 
static int TrainAndEvaluate(std::string mnist_train_path, 
                            std::string mnist_test_path);

//评估 
static double Evaluate(std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& mnist_test_sample_data_set, 
                       std::vector<std::vector<std::vector<double>>>& mnist_test_label_data_set);

//训练和评估  学习策略
int TrainAndEvaluate(std::string mnist_train_path, 
                     std::string mnist_test_path) {
    //得到训练数据集
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> mnist_training_sample_data_set; 
    std::vector<std::vector<std::vector<double>>> mnist_training_label_data_set;
    if (-1 == GetMnistTrainingDataSet(mnist_train_path, FLAGS_mnist_training_data_size, 
                                      mnist_training_sample_data_set, 
                                      mnist_training_label_data_set)) {
        LOG(ERROR) << "get mnist traing data set failed...";
        return -1;
    }

    //得到测试数据集
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> mnist_test_sample_data_set; 
    std::vector<std::vector<std::vector<double>>> mnist_test_label_data_set;
    if (-1 == GetMnistTestDataSet(mnist_test_path, FLAGS_mnist_test_data_size, 
                                  mnist_test_sample_data_set, 
                                  mnist_test_label_data_set)) {
        LOG(ERROR) << "get mnist traing data set failed...";
        return -1;
    }

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
                                                    mnist_training_sample_data_set[0])) {
        LOG(ERROR) << "LeNet-5 initializer failed, programm is exiting";
        return -1;
    }

    //记录上次的测试错误率 如果本次测试错误率高于上次 证明模型过拟合了 退出训练   
    double last_error_ratio = 1.0;

    LOG(INFO) << "=================开始训练==================";
    while (true) {
        FLAGS_epoch++;
        //每次训练完成1轮后 打印一下当前情况
        if (-1 == SingletonLeNet::Instance().Train(mnist_training_sample_data_set, 
                                                   mnist_training_label_data_set, 
                                                   FLAGS_learning_rate, 
                                                   FLAGS_batch_size)) {
            LOG(ERROR) << "LeNet-5 training and evaluate failed";
            return -1;
        }
        if (FLAGS_stop_flag) {
            break;
        }
        return 0;
        //得到当前时间
        char now[60] = { 0 };
        calculate::time::GetCurrentTime(now);
        //拿一个样本和标签 算loss
        std::vector<std::vector<double>> output_array;
        SingletonLeNet::Instance().Predict(mnist_test_sample_data_set[0], 
                                           output_array);
        double loss = SingletonLeNet::Instance().Loss(output_array, 
                                                      mnist_test_label_data_set[0]);
        LOG(INFO) << now << " epoch " << FLAGS_epoch
                  << " finished, loss: " << loss;
        
        //每训练1轮 测试一次
        if (0 == FLAGS_epoch % 1) {
            double current_error_ratio = Evaluate(mnist_test_sample_data_set, 
                                                  mnist_test_label_data_set);
            char _now[60] = { 0 };
            calculate::time::GetCurrentTime(_now);
            LOG(WARNING) << _now << " after epoch " << FLAGS_epoch
                         << ", error ratio is: %" << current_error_ratio * 100;
            if (current_error_ratio > last_error_ratio) {
                LOG(INFO) << "=================训练结束==================";
                break;
            } else {
                last_error_ratio = current_error_ratio;
            }

            //学习率减半
            FLAGS_learning_rate /= 2.0;
        }
    }

    return 0;
}

//评估 
double Evaluate(std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& mnist_test_sample_data_set, 
                std::vector<std::vector<std::vector<double>>>& mnist_test_label_data_set) {
    double error = 0.0;
    size_t test_data_set_total = mnist_test_sample_data_set.size();
    int label = -1;
    int output = -1;
    std::vector<std::vector<double>> output_array;
    //遍历测试数据集 做预测 查看结果 
    for (int i = 0; i < test_data_set_total; i++) {
        label = GetPredictResult(mnist_test_label_data_set[i]);
        SingletonLeNet::Instance().Predict(mnist_test_sample_data_set[i], 
                                           output_array);
        output = GetPredictResult(output_array);
        //LOG(INFO) << "预测值: " << output << ", 标签: " << label;
        if (label != output) {
            error += 1.0;
        }
    }
    
    return error / test_data_set_total;
}


int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "google_logging/");
    FLAGS_stderrthreshold = 0;
    FLAGS_colorlogtostderr = true;
    
    //注册信号处理函数
    struct sigaction signal_action;
    signal_action.sa_sigaction = SignalAction;   //捕获信号后调用的函数 
    signal_action.sa_flags = SA_SIGINFO;         //flags是SIGINFO时 sa_sigaction作为callback 否则是handle作为callback
    sigaction(SIGINT,  &signal_action, nullptr); //注册信号处理回调 sigint对应的是ctrl + c
    sigaction(SIGTERM, &signal_action, nullptr);
    sigaction(SIGUSR1, &signal_action, nullptr);
    sigaction(SIGUSR2, &signal_action, nullptr);

    //记录开始时间
    std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();

    std::string mnist_path = "./data/";
    std::string mnist_train_path = mnist_path + "mnist_training_set/";
    std::string mnist_test_path = mnist_path + "mnist_test_set/";
#if GPU 
    if (!calculate::cuda::InitializeCUDA()) {
        LOG(ERROR) << "initialize CUDA failed";
        google::ShutdownGoogleLogging();
        return 0;
    } else {
        LOG(INFO) << "successfully initialize CUDA!";
        calculate::cuda::GpuInfoShow();
    }
#endif
    TrainAndEvaluate(mnist_train_path, mnist_test_path);

    //SingletonLeNet::Instance().DumpModel(FLAGS_lenet_weights_output_file);
    
    //记录结束时间
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    //设置单位为秒
    //std::chrono::duration<int, std::ratio<1, 1>> sec = std::chrono::duration_cast<
    //                                                   std::chrono::seconds>(end - begin);
    std::chrono::duration<int, std::milli> milli = std::chrono::duration_cast<
                                                   std::chrono::milliseconds>(end - begin);
    //打印耗时
    //LOG(INFO) << "programming is exiting, the total time is: " << sec.count() << "s";
    LOG(INFO) << "programming is exiting, the total time is: " << milli.count() << "ms";
    google::ShutdownGoogleLogging();
    
    return 0;

}


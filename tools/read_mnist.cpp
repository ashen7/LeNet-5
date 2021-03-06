/*
 * =====================================================================================
 *
 *       Filename:  read_mnist.cpp
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
#include "read_mnist.h"

#include <time.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <atomic>

#include <glog/logging.h>

#include "utility/normalizer.hpp"

//得到训练集
int GetMnistTrainingDataSet(std::string mnist_train_path, 
                            size_t trainging_picture_number, 
                            std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& mnist_sample_data_set, 
                            std::vector<std::vector<std::vector<double>>>& mnist_label_data_set) {
    std::string mnist_trainging_sample = mnist_train_path + "train-images-idx3-ubyte";
    std::string mnist_trainging_label = mnist_train_path + "train-labels-idx1-ubyte";
    
    if (-1 == LoadMnistImage(mnist_trainging_sample, 
                             trainging_picture_number, 
                             mnist_sample_data_set)) {
        LOG(ERROR) << "load mnist training sample failed...";
        return -1;
    }

    if (-1 == LoadMnistLabel(mnist_trainging_label, 
                             trainging_picture_number,
                             mnist_label_data_set)) {
        LOG(ERROR) << "load mnist training label failed...";
        return -1;
    }

    LOG(INFO) << "successfully load mnist training data set, load picture: " 
              << trainging_picture_number;
    return 0;
}

//得到测试集
int GetMnistTestDataSet(std::string mnist_test_path, 
                        size_t test_picture_number, 
                        std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& mnist_sample_data_set, 
                        std::vector<std::vector<std::vector<double>>>& mnist_label_data_set) {
    std::string mnist_test_sample = mnist_test_path + "t10k-images-idx3-ubyte";
    std::string mnist_test_label = mnist_test_path + "t10k-labels-idx1-ubyte";
    
    if (-1 == LoadMnistImage(mnist_test_sample, 
                             test_picture_number, 
                             mnist_sample_data_set)) {
        LOG(ERROR) << "load mnist test sample failed...";
        return -1;
    }

    if (-1 == LoadMnistLabel(mnist_test_label, 
                             test_picture_number,
                             mnist_label_data_set)) {
        LOG(ERROR) << "load mnist test label failed...";
        return -1;
    }

    LOG(INFO) << "successfully load mnist test data set, load picture: "
              << test_picture_number;
    return 0;
}

//导入样本
int LoadMnistImage(std::string mnist_image_file, 
                   size_t sample_picture_number, 
                   std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& mnist_sample_data_set) {
    std::ifstream mnist_image;
    mnist_image.open(mnist_image_file.c_str(), std::ios::in | std::ios::binary);
    if (!mnist_image.is_open()) {
        LOG(ERROR) << "open file failed, filename is :" << mnist_image_file.c_str();
        return -1;
    }

    //shared_ptr存数组类型 要自己写删除器
    size_t mnist_image_data_size = 16 + sample_picture_number * kMnistImageSize;
    std::shared_ptr<uint8_t> mnist_image_data(new uint8_t[mnist_image_data_size], [](uint8_t* data) {
                                              delete []data; });
    //把数据读入智能指针中
    while (!mnist_image.eof()) {
        mnist_image.read(reinterpret_cast<char*>(mnist_image_data.get()), mnist_image_data_size);
    }
    mnist_image.close();
    
    if (0 != mnist_sample_data_set.size()) {
        mnist_sample_data_set.clear();
    }

    //最后都加入到这个数组中去
    mnist_sample_data_set.reserve(sample_picture_number);
    for (int i = 0; i < sample_picture_number; i++) {
        //保存每一张图像
        std::vector<std::vector<std::vector<uint8_t>>> image_data;
        GetOneImageData(mnist_image_data, i, image_data);
        mnist_sample_data_set.push_back(image_data);
    }

    return 0;
}

//得到一个样本 图片是28*28 
void GetOneImageData(const std::shared_ptr<uint8_t> mnist_image_data, 
                     size_t current_picture_count, 
                     std::vector<std::vector<std::vector<uint8_t>>>& image_data) {
    if (0 == image_data.size()) {
        image_data = std::vector<std::vector<std::vector<uint8_t>>>(kMnistImageChannelNumber,
                                                                    std::vector<std::vector<uint8_t>>(kMnistImageHeight, 
                                                                    std::vector<uint8_t>(kMnistImageWidth, 0)));
    }
    
    //图片数据 从索引16开始 16 + 28 * 28是第一张图片 16 + 28 * 28 + 28 * 28是第二张
    size_t start_index = 16 + current_picture_count * kMnistImageSize;
    for (int i = 0; i < kMnistImageChannelNumber; i++) {
        for (int j = 0; j < kMnistImageHeight; j++) {
            for (int k = 0; k < kMnistImageWidth; k++) {
                image_data[i][j][k] = mnist_image_data.get()[start_index++];
            }
        }
    }
}

//导入标签
int LoadMnistLabel(std::string mnist_label_file,  
                   size_t label_picture_number, 
                   std::vector<std::vector<std::vector<double>>>& mnist_label_data_set) {
    std::ifstream mnist_label;
    mnist_label.open(mnist_label_file.c_str(), std::ios::in | std::ios::binary);
    if (!mnist_label.is_open()) {
        LOG(ERROR) << "open file failed, filename is :" << mnist_label_file.c_str();
        return -1;
    }

    //shared_ptr存数组类型 要自己写删除器
    size_t mnist_label_data_size = 8 + label_picture_number * sizeof(char);
    std::shared_ptr<uint8_t> mnist_label_data(new uint8_t[mnist_label_data_size], [](uint8_t* data) {
                                              delete []data; });
    //把数据读入指针指针中
    while (!mnist_label.eof()) {
        mnist_label.read(reinterpret_cast<char*>(mnist_label_data.get()), mnist_label_data_size);
    }
    mnist_label.close();
    
    if (0 != mnist_label_data_set.size()) {
        mnist_label_data_set.clear();
    }

    //最后都加入到这个数组中去
    mnist_label_data_set.reserve(label_picture_number);
    for (int i = 0; i < label_picture_number; i++) {
        //从索引8开始 每个值是样本对应的label值
        uint8_t label_value = mnist_label_data.get()[8 + i];
        //保存每一张图像
        std::vector<std::vector<double>> label_data;
        size_t label_data_rows = kOutputNode;
        size_t label_data_cols = 1;

        utility::Normalizer::Normalize(label_value, label_data_rows, label_data_cols, label_data);
        mnist_label_data_set.push_back(label_data);
    }

    return 0;
}

//10行1列 每个值就是一个类别的预测值 取最大的值就是预测结果
int GetPredictResult(const std::vector<std::vector<double>>& output_array) {
    double max_value = 0.0; 
    int max_value_index = 0;
    for (int i = 0; i < output_array.size(); i++) {
        for (int j = 0; j < output_array[i].size(); j++) {
            if (output_array[i][j] > max_value) {
                max_value = output_array[i][j];
                max_value_index = i;
            }
        }
    }
    
    return max_value_index;
}

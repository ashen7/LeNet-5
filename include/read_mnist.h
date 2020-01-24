#ifndef READ_MNIST_H_
#define READ_MNIST_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <atomic>

//global variable
const size_t kMnistImageHeight = 28;
const size_t kMnistImageWidth = 28;
const size_t kMnistImageChannelNumber = 1;
const size_t kMnistImageSize = kMnistImageHeight * kMnistImageWidth * kMnistImageChannelNumber;
const size_t kOutputNode = 10;

//得到训练集
int GetMnistTrainingDataSet(std::string mnist_train_path, 
                            size_t trainging_picture_number, 
                            std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& mnist_sample_data_set, 
                            std::vector<std::vector<std::vector<double>>>& mnist_label_data_set);
            
//得到测试集
int GetMnistTestDataSet(std::string mnist_test_path, 
                        size_t test_picture_number, 
                        std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& mnist_sample_data_set, 
                        std::vector<std::vector<std::vector<double>>>& mnist_label_data_set);

//导入样本
int LoadMnistImage(std::string mnist_image_file, 
                   size_t sample_picture_number, 
                   std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& mnist_sample_data_set);

//导入标签
int LoadMnistLabel(std::string mnist_label_file, 
                   size_t label_picture_number, 
                   std::vector<std::vector<std::vector<double>>>& mnist_label_data_set);

//得到一个样本
void GetOneImageData(const std::shared_ptr<uint8_t> mnist_image_data, 
                     size_t current_image_number, 
                     std::vector<std::vector<std::vector<uint8_t>>>& image_data);

//得到结果
int GetPredictResult(const std::vector<std::vector<double>>& output_array);


#endif   //READ_MNIST_H_

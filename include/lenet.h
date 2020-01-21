/*
 * =====================================================================================
 *
 *       Filename:  lenet.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2019年12月29日 19时38分17秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef CNN_LENET_H_
#define CNN_LENET_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>
#include <functional>

#include "utility/singleton.hpp"

namespace dnn {
//类的前置声明 神经网络(1-3层全连接层)
class NeuralNetwork;
}     //namespace dnn

namespace cnn {
//类的前置声明 卷积层 和 池化层
class ConvolutionalLayer;   
class MaxPoolingLayer;

//LeNet-5模型
class LeNet {
public:
    typedef std::vector<double> Matrix1d;
    typedef std::vector<std::vector<double>> Matrix2d;
    typedef std::vector<std::vector<std::vector<double>>> Matrix3d;
    typedef std::vector<std::vector<std::vector<std::vector<double>>>> Matrix4d;
    typedef std::vector<uint8_t> ImageMatrix1d;
    typedef std::vector<std::vector<uint8_t>> ImageMatrix2d;
    typedef std::vector<std::vector<std::vector<uint8_t>>> ImageMatrix3d;
    typedef std::vector<std::vector<std::vector<std::vector<uint8_t>>>> ImageMatrix4d;

    LeNet();
    ~LeNet();
    LeNet(const LeNet&) = delete;
    LeNet& operator =(const LeNet&) = delete;
    LeNet(LeNet&&) = default;
    LeNet& operator =(LeNet&&) = default;

    const Matrix2d& get_output_array() const noexcept {
        return output_array_;
    }

    const Matrix3d& get_delta_array() const noexcept {
        return delta_array_;
    }
    
public:
    int Initialize(int conv1_input_height,  int conv1_input_width,  int conv1_channel_number, 
                   int conv1_filter_height, int conv1_filter_width, int conv1_filter_number, 
                   int conv1_zero_padding,  int conv1_stride, 
                   int pool2_filter_height, int pool2_filter_width, int pool2_stride, 
                   int conv3_filter_height, int conv3_filter_width, int conv3_filter_number, 
                   int conv3_zero_padding,  int conv3_stride, 
                   int pool4_filter_height, int pool4_filter_width, int pool4_stride, 
                   int fc5_output_node,     int fc6_output_node,    double learning_rate, 
                   const Matrix3d& sample);
    int Initialize(int conv1_input_height,  int conv1_input_width,  int conv1_channel_number, 
                   int conv1_filter_height, int conv1_filter_width, int conv1_filter_number, 
                   int conv1_zero_padding,  int conv1_stride, 
                   int pool2_filter_height, int pool2_filter_width, int pool2_stride, 
                   int conv3_filter_height, int conv3_filter_width, int conv3_filter_number, 
                   int conv3_zero_padding,  int conv3_stride, 
                   int pool4_filter_height, int pool4_filter_width, int pool4_stride, 
                   int fc5_output_node,     int fc6_output_node,    double learning_rate, 
                   const ImageMatrix3d& sample);
    int Train(const Matrix4d& training_sample, const Matrix3d& training_label, int batch_size);
    int Train(const ImageMatrix4d& training_sample, const Matrix3d& training_label, int batch_size);
    int Predict(const Matrix3d& input_array, Matrix2d& output_array);
    int Predict(const ImageMatrix3d& input_array, Matrix2d& output_array);
    int CalcGradient(const Matrix2d& output_array, Matrix3d& delta_array);
    int Forward(const Matrix3d& input_array);
    int Forward(const ImageMatrix3d& input_array);
    int Backward(const Matrix2d& label); 
    void UpdateWeights(double learning_rate, int batch_size);
    double Loss(const Matrix2d& output_array, const Matrix2d& label) const noexcept;
    //void Dump() const noexcept;

protected:
    //内部函数
    int InitLeNetModel(const Matrix3d& sample);
    int InitLeNetModel(const ImageMatrix3d& sample);
    int TrainOneSample(const Matrix3d& sample, const Matrix2d& label);
    int TrainOneSample(const ImageMatrix3d& sample, const Matrix2d& label);

private:
    int conv1_input_height_;     //构造第一层 卷积层的参数
    int conv1_input_width_;
    int conv1_channel_number_;
    int conv1_filter_height_;
    int conv1_filter_width_;
    int conv1_filter_number_;
    int conv1_zero_padding_;
    int conv1_stride_;

    int pool2_filter_height_;    //构造第二层 池化层的参数
    int pool2_filter_width_;
    int pool2_stride_;

    int conv3_filter_height_;    //构造第三层 卷积层的参数
    int conv3_filter_width_;
    int conv3_filter_number_;
    int conv3_zero_padding_;
    int conv3_stride_;

    int pool4_filter_height_;    //构造第四层 池化层的参数
    int pool4_filter_width_;
    int pool4_stride_;

    int fc5_input_node_;         //构造第5层 全连接层的参数
    int fc5_output_node_;        //构造第5层 全连接层的参数
    int fc6_output_node_;        //构造第6层 全连接层的参数(输出层)

    double learning_rate_;       //学习率
    
    int pool4_output_depth_;     //记录第四层 池化层输出的深度 高 宽 reshape时使用
    int pool4_output_height_;
    int pool4_output_width_;

    Matrix2d output_array_;      //输出数组
    Matrix3d delta_array_;       //误差数组

    std::shared_ptr<ConvolutionalLayer> convolutional_layer1_;  //第一层 卷积层
    std::shared_ptr<MaxPoolingLayer>    max_pooling_layer2_;    //第二层 池化层
    std::shared_ptr<ConvolutionalLayer> convolutional_layer3_;  //第三层 卷积层
    std::shared_ptr<MaxPoolingLayer>    max_pooling_layer4_;    //第四层 池化层
    std::shared_ptr<dnn::NeuralNetwork> neural_network_layer5_; //第五层 神经网络(1-3层全连接层)
};


}      //namespace cnn

//单例模式
typedef typename utility::Singleton<cnn::LeNet> SingletonLeNet;

#endif //CNN_LENET_H_

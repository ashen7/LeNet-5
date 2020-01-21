/*
 * =====================================================================================
 *
 *       Filename:  lenet.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2020年01月04日 10时44分21秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "lenet.h"
#include "convolutional_layer.h"
#include "max_pooling_layer.h"
#include "filter.h"
#include "neural_network.h"
#include "full_connected_layer.h"

#include <memory>
#include <vector>
#include <functional>

#include <glog/logging.h>

#include "utility/matrix_math_function.hpp"

namespace cnn {

LeNet::LeNet() {
    //给每层new一个对象 调用默认的无参构造函数
    convolutional_layer1_ = std::make_shared<ConvolutionalLayer>();
    max_pooling_layer2_ = std::make_shared<MaxPoolingLayer>();
    convolutional_layer3_ = std::make_shared<ConvolutionalLayer>();
    max_pooling_layer4_ = std::make_shared<MaxPoolingLayer>();
    neural_network_layer5_ = std::make_shared<dnn::NeuralNetwork>(); 
}

LeNet::~LeNet() {
    //释放每层资源
    convolutional_layer1_.reset();
    max_pooling_layer2_.reset();
    convolutional_layer3_.reset();
    max_pooling_layer4_.reset();
    neural_network_layer5_.reset();
}

/*
 * 初始化LeNet-5
 * 第一层卷积层   输入的宽高深度 filter的宽高个数 补0填充 步长
 * 第二层池化层   filter的宽高(深度不变) 步长
 * 第三层卷积层   filter的宽高个数 补0填充 步长
 * 第四层池化层   filter的宽高(深度不变) 步长
 * 第五层全连接层 输出节点数量
 * 第六层全连接层 输出节点数量(输出层)
 * 学习率
 * 一个样本(输入特征) 用来计算一次前向传播 来初始化LeNet每层
 */
int LeNet::Initialize(int conv1_input_height,  int conv1_input_width,  int conv1_channel_number, 
                      int conv1_filter_height, int conv1_filter_width, int conv1_filter_number, 
                      int conv1_zero_padding,  int conv1_stride, 
                      int pool2_filter_height, int pool2_filter_width, int pool2_stride, 
                      int conv3_filter_height, int conv3_filter_width, int conv3_filter_number, 
                      int conv3_zero_padding,  int conv3_stride, 
                      int pool4_filter_height, int pool4_filter_width, int pool4_stride,
                      int fc5_output_node,     int fc6_output_node,    double learning_rate, 
                      const Matrix3d& sample) {
    if (conv1_input_height <= 0
            || conv1_input_width <= 0
            || conv1_channel_number <= 0
            || conv1_filter_height <= 0
            || conv1_filter_width <= 0
            || conv1_filter_number <= 0
            || conv1_zero_padding < 0
            || conv1_stride <= 0
            || pool2_filter_height <= 0
            || pool2_filter_width <= 0
            || pool2_stride <= 0 
            || conv3_filter_height <= 0
            || conv3_filter_width <= 0
            || conv3_filter_number <= 0
            || conv3_zero_padding < 0
            || conv3_stride <= 0
            || pool4_filter_height <= 0
            || pool4_filter_width <= 0
            || pool4_stride <= 0
            || fc5_output_node <= 0
            || fc6_output_node <= 0
            || learning_rate <= 0.0) {
        LOG(ERROR) << "LeNet-5 initialize failed, input parameter is wrong";
        return -1;
    }

    conv1_input_height_ = conv1_input_height;
    conv1_input_width_ = conv1_input_width;
    conv1_channel_number_ = conv1_channel_number;
    conv1_filter_height_ = conv1_filter_height;
    conv1_filter_width_ = conv1_filter_width;
    conv1_filter_number_ = conv1_filter_number;
    conv1_zero_padding_ = conv1_zero_padding;
    conv1_stride_ = conv1_stride;

    pool2_filter_height_ = pool2_filter_height;
    pool2_filter_width_ = pool2_filter_width;
    pool2_stride_ = pool2_stride;

    conv3_filter_height_ = conv3_filter_height;
    conv3_filter_width_ = conv3_filter_width;
    conv3_filter_number_ = conv3_filter_number;
    conv3_zero_padding_ = conv3_zero_padding;
    conv3_stride_ = conv3_stride;

    pool4_filter_height_ = pool4_filter_height;
    pool4_filter_width_ = pool4_filter_width;
    pool4_stride_ = pool4_stride;

    fc5_output_node_ = fc5_output_node;
    fc6_output_node_ = fc6_output_node;

    learning_rate_ = learning_rate;
    
    //先调用第一层卷积层的初始化 卷积层会初始化权重数组和偏置
    //后面算出本层的输入 得到输出数组的shape 就能去初始化下一层 依次
    if (-1 == convolutional_layer1_->Initialize(conv1_input_height, conv1_input_width, conv1_channel_number,
                                                conv1_filter_height, conv1_filter_width, conv1_filter_number, 
                                                conv1_zero_padding, conv1_stride)) {
        LOG(ERROR) << "LeNet-5 initialize failed, convolutional layer 1 input parameter is wrong";
        return -1;
    }

    //调用InitLeNetModel 初始化LeNet所有层 通过前向计算一层层得到输出 传入下一层来初始化
    if (-1 == InitLeNetModel(sample)) {
        LOG(ERROR) << "LeNet-5 initialize failed, Init LeNet Model occur error";
        return -1;
    }

    return 0;
}

/*
 * 初始化LeNet-5
 * 第一层卷积层   输入的宽高深度 filter的宽高个数 补0填充 步长
 * 第二层池化层   filter的宽高(深度不变) 步长
 * 第三层卷积层   filter的宽高个数 补0填充 步长
 * 第四层池化层   filter的宽高(深度不变) 步长
 * 第五层全连接层 输出节点数量
 * 第六层全连接层 输出节点数量(输出层)
 * 学习率
 * 一个样本(输入特征) 用来计算一次前向传播 来初始化LeNet每层
 */
int LeNet::Initialize(int conv1_input_height,  int conv1_input_width,  int conv1_channel_number, 
                      int conv1_filter_height, int conv1_filter_width, int conv1_filter_number, 
                      int conv1_zero_padding,  int conv1_stride, 
                      int pool2_filter_height, int pool2_filter_width, int pool2_stride, 
                      int conv3_filter_height, int conv3_filter_width, int conv3_filter_number, 
                      int conv3_zero_padding,  int conv3_stride, 
                      int pool4_filter_height, int pool4_filter_width, int pool4_stride,
                      int fc5_output_node,     int fc6_output_node,    double learning_rate, 
                      const ImageMatrix3d& sample) {
    if (conv1_input_height <= 0
            || conv1_input_width <= 0
            || conv1_channel_number <= 0
            || conv1_filter_height <= 0
            || conv1_filter_width <= 0
            || conv1_filter_number <= 0
            || conv1_zero_padding < 0
            || conv1_stride <= 0
            || pool2_filter_height <= 0
            || pool2_filter_width <= 0
            || pool2_stride <= 0 
            || conv3_filter_height <= 0
            || conv3_filter_width <= 0
            || conv3_filter_number <= 0
            || conv3_zero_padding < 0
            || conv3_stride <= 0
            || pool4_filter_height <= 0
            || pool4_filter_width <= 0
            || pool4_stride <= 0
            || fc5_output_node <= 0
            || fc6_output_node <= 0
            || learning_rate <= 0.0) {
        LOG(ERROR) << "LeNet-5 initialize failed, input parameter is wrong";
        return -1;
    }

    conv1_input_height_ = conv1_input_height;
    conv1_input_width_ = conv1_input_width;
    conv1_channel_number_ = conv1_channel_number;
    conv1_filter_height_ = conv1_filter_height;
    conv1_filter_width_ = conv1_filter_width;
    conv1_filter_number_ = conv1_filter_number;
    conv1_zero_padding_ = conv1_zero_padding;
    conv1_stride_ = conv1_stride;

    pool2_filter_height_ = pool2_filter_height;
    pool2_filter_width_ = pool2_filter_width;
    pool2_stride_ = pool2_stride;

    conv3_filter_height_ = conv3_filter_height;
    conv3_filter_width_ = conv3_filter_width;
    conv3_filter_number_ = conv3_filter_number;
    conv3_zero_padding_ = conv3_zero_padding;
    conv3_stride_ = conv3_stride;

    pool4_filter_height_ = pool4_filter_height;
    pool4_filter_width_ = pool4_filter_width;
    pool4_stride_ = pool4_stride;

    fc5_output_node_ = fc5_output_node;
    fc6_output_node_ = fc6_output_node;

    learning_rate_ = learning_rate;
    
    //先调用第一层卷积层的初始化 卷积层会初始化权重数组和偏置
    //后面算出本层的输入 得到输出数组的shape 就能去初始化下一层 依次
    if (-1 == convolutional_layer1_->Initialize(conv1_input_height, conv1_input_width, conv1_channel_number,
                                                conv1_filter_height, conv1_filter_width, conv1_filter_number, 
                                                conv1_zero_padding, conv1_stride)) {
        LOG(ERROR) << "LeNet-5 initialize failed, convolutional layer 1 input parameter is wrong";
        return -1;
    }

    //调用InitLeNetModel 初始化LeNet所有层 通过前向计算一层层得到输出 传入下一层来初始化
    if (-1 == InitLeNetModel(sample)) {
        LOG(ERROR) << "LeNet-5 initialize failed, Init LeNet Model occur error";
        return -1;
    }

    return 0;
}



/*
 * 初始化LeNet模型每层 内部函数 
 * 传入一个样本 计算前向传播 一层层得到本层输出 作为下一层输入用来初始化
 */
int LeNet::InitLeNetModel(const ImageMatrix3d& input_array) { 
    //1. 计算卷积层1的前向计算结果 补0填充2 输入28*28*1变32*32*1 filter5*5*32 步长1 得到28*28*32输出特征图
    if (-1 == convolutional_layer1_->Forward(input_array)) {
        LOG(ERROR) << "Init LeNet Model failed, conv layer 1 forward occur error";
        return -1;
    }

    //得到第一层卷积层的输出 初始化第二层池化层 
    auto conv1_output_shape = Matrix::GetShape(convolutional_layer1_->get_output_array());
    int conv1_output_depth;
    int conv1_output_height;
    int conv1_output_width;
    std::tie(conv1_output_depth, conv1_output_height, conv1_output_width) = conv1_output_shape;
    if (conv1_output_depth <=0 
            || conv1_output_height <= 0
            || conv1_output_width <=0) {
        LOG(ERROR) << "Init LeNet Model failed, conv layer 1 forward occur error";
        return -1;
    }
    if (-1 == max_pooling_layer2_->Initialize(conv1_output_height, conv1_output_width, 
                                              conv1_output_depth,  pool2_filter_height_, 
                                              pool2_filter_width_, pool2_stride_)) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 2 initialize occur error";
        return -1;
    }

    //2. 计算池化层2的前向计算结果 输入28*28*32 filter2*2 步长2 下采样降维输出14*14*32
    if (-1 == max_pooling_layer2_->Forward(convolutional_layer1_->get_output_array())) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 2 forward occur error";
        return -1;
    }

    //得到第二层池化层的输出 初始化第三层卷积层
    auto pool2_output_shape = Matrix::GetShape(max_pooling_layer2_->get_output_array());
    int pool2_output_depth;
    int pool2_output_height;
    int pool2_output_width;
    std::tie(pool2_output_depth, pool2_output_height, pool2_output_width) = pool2_output_shape;
    if (pool2_output_depth <=0 
            || pool2_output_height <= 0
            || pool2_output_width <=0) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 2 forward occur error";
        return -1;
    }
    if (-1 == convolutional_layer3_->Initialize(pool2_output_height, pool2_output_width, 
                                                pool2_output_depth,  conv3_filter_height_, 
                                                conv3_filter_width_, conv3_filter_number_, 
                                                conv3_zero_padding_, conv3_stride_)) {
        LOG(ERROR) << "Init LeNet Model failed, conv layer 3 initialize occur error";
        return -1;
    }

    //3. 计算卷积层3的前向计算结果 补0填充2 输入14*14*32变18*18*32 filter5*5*64 步长1 得到14*14*64输出特征图
    if (-1 == convolutional_layer3_->Forward(max_pooling_layer2_->get_output_array())) {
        LOG(ERROR) << "Init LeNet Model failed, conv layer 3 forward occur error";
        return -1;
    }

    //得到第三层卷积层的输出 初始化第四层池化层
    auto conv3_output_shape = Matrix::GetShape(convolutional_layer3_->get_output_array());
    int conv3_output_depth;
    int conv3_output_height;
    int conv3_output_width;
    std::tie(conv3_output_depth, conv3_output_height, conv3_output_width) = conv3_output_shape;
    if (conv3_output_depth <=0 
            || conv3_output_height <= 0
            || conv3_output_width <=0) {
        LOG(ERROR) << "Init LeNet Model failed, conv layer 3 forward occur error";
        return -1;
    }
    if (-1 == max_pooling_layer4_->Initialize(conv3_output_height, conv3_output_width, 
                                              conv3_output_depth,  pool4_filter_height_, 
                                              pool4_filter_width_, pool4_stride_)) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 4 initialize occur error";
        return -1;
    }

    //4. 计算池化层4的前向计算结果 输入14*14*64 filter2*2 步长2 下采样降维输出7*7*64
    if (-1 == max_pooling_layer4_->Forward(convolutional_layer3_->get_output_array())) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 4 forward occur error";
        return -1;
    } 

    //得到第四层池化层的输出 初始化第五层神经网络(1-3层全连接层) 
    //输入7*7*64=3136*1列向量  输出为512*1列向量 
    auto pool4_output_shape = Matrix::GetShape(max_pooling_layer4_->get_output_array());
    std::tie(pool4_output_depth_, pool4_output_height_, pool4_output_width_) = pool4_output_shape;
    if (pool4_output_depth_ <=0 
            || pool4_output_height_ <= 0
            || pool4_output_width_ <= 0) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 4 forward occur error";
        return -1;
    }
    fc5_input_node_ = pool4_output_depth_ * pool4_output_height_ * pool4_output_width_;
    std::vector<size_t> fc_layers_node_array{fc5_input_node_, fc5_output_node_, fc6_output_node_};
    if (-1 == neural_network_layer5_->Initialize(fc_layers_node_array)) {
        LOG(ERROR) << "Init LeNet Model failed, full connected layer 5 initialize occur error";
        return -1;
    }

    //5. 计算全连接层5的前向计算结果和全连接层6的前向计算结果 这里第4层的输出reshape成3136*1的列向量 送入fc 
    Matrix2d fc5_input_array;
    Matrix::Reshape(max_pooling_layer4_->get_output_array(), fc5_input_node_, 1, fc5_input_array);
    if (-1 == neural_network_layer5_->Predict(fc5_input_array, output_array_)) {
        LOG(ERROR) << "Init LeNet Model failed, full connected layer 5 forward occur error";
        return -1;
    }
    
    return 0;
}

/*
 * 初始化LeNet模型每层 内部函数 
 * 传入一个样本 计算前向传播 一层层得到输出 作为下一层输入初始化
 */
int LeNet::InitLeNetModel(const Matrix3d& input_array) { 
    //1. 计算卷积层1的前向计算结果 补0填充2 输入28*28*1变32*32*1 filter5*5*32 步长1 得到28*28*32输出特征图
    if (-1 == convolutional_layer1_->Forward(input_array)) {
        LOG(ERROR) << "Init LeNet Model failed, conv layer 1 forward occur error";
        return -1;
    }

    //得到第一层卷积层的输出 初始化第二层池化层 
    auto conv1_output_shape = Matrix::GetShape(convolutional_layer1_->get_output_array());
    int conv1_output_depth;
    int conv1_output_height;
    int conv1_output_width;
    std::tie(conv1_output_depth, conv1_output_height, conv1_output_width) = conv1_output_shape;
    if (conv1_output_depth <=0 
            || conv1_output_height <= 0
            || conv1_output_width <=0) {
        LOG(ERROR) << "Init LeNet Model failed, conv layer 1 forward occur error";
        return -1;
    }
    if (-1 == max_pooling_layer2_->Initialize(conv1_output_height, conv1_output_width, 
                                              conv1_output_depth,  pool2_filter_height_, 
                                              pool2_filter_width_, pool2_stride_)) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 2 initialize occur error";
        return -1;
    }

    //2. 计算池化层2的前向计算结果 输入28*28*32 filter2*2 步长2 下采样降维输出14*14*32
    if (-1 == max_pooling_layer2_->Forward(convolutional_layer1_->get_output_array())) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 2 forward occur error";
        return -1;
    }

    //得到第二层池化层的输出 初始化第三层卷积层
    auto pool2_output_shape = Matrix::GetShape(max_pooling_layer2_->get_output_array());
    int pool2_output_depth;
    int pool2_output_height;
    int pool2_output_width;
    std::tie(pool2_output_depth, pool2_output_height, pool2_output_width) = pool2_output_shape;
    if (pool2_output_depth <=0 
            || pool2_output_height <= 0
            || pool2_output_width <=0) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 2 forward occur error";
        return -1;
    }
    if (-1 == convolutional_layer3_->Initialize(pool2_output_height, pool2_output_width, 
                                                pool2_output_depth,  conv3_filter_height_, 
                                                conv3_filter_width_, conv3_filter_number_, 
                                                conv3_zero_padding_, conv3_stride_)) {
        LOG(ERROR) << "Init LeNet Model failed, conv layer 3 initialize occur error";
        return -1;
    }

    //3. 计算卷积层3的前向计算结果 补0填充2 输入14*14*32变18*18*32 filter5*5*64 步长1 得到14*14*64输出特征图
    if (-1 == convolutional_layer3_->Forward(max_pooling_layer2_->get_output_array())) {
        LOG(ERROR) << "Init LeNet Model failed, conv layer 3 forward occur error";
        return -1;
    }

    //得到第三层卷积层的输出 初始化第四层池化层
    auto conv3_output_shape = Matrix::GetShape(convolutional_layer3_->get_output_array());
    int conv3_output_depth;
    int conv3_output_height;
    int conv3_output_width;
    std::tie(conv3_output_depth, conv3_output_height, conv3_output_width) = conv3_output_shape;
    if (conv3_output_depth <=0 
            || conv3_output_height <= 0
            || conv3_output_width <=0) {
        LOG(ERROR) << "Init LeNet Model failed, conv layer 3 forward occur error";
        return -1;
    }
    if (-1 == max_pooling_layer4_->Initialize(conv3_output_height, conv3_output_width, 
                                              conv3_output_depth,  pool4_filter_height_, 
                                              pool4_filter_width_, pool4_stride_)) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 4 initialize occur error";
        return -1;
    }

    //4. 计算池化层4的前向计算结果 输入14*14*64 filter2*2 步长2 下采样降维输出7*7*64
    if (-1 == max_pooling_layer4_->Forward(convolutional_layer3_->get_output_array())) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 4 forward occur error";
        return -1;
    } 

    //得到第四层池化层的输出 初始化第五层神经网络(1-3层全连接层) 
    //输入7*7*64=3136*1列向量  输出为512*1列向量 
    auto pool4_output_shape = Matrix::GetShape(max_pooling_layer4_->get_output_array());
    std::tie(pool4_output_depth_, pool4_output_height_, pool4_output_width_) = pool4_output_shape;
    if (pool4_output_depth_ <=0 
            || pool4_output_height_ <= 0
            || pool4_output_width_ <= 0) {
        LOG(ERROR) << "Init LeNet Model failed, pool layer 4 forward occur error";
        return -1;
    }
    fc5_input_node_ = pool4_output_depth_ * pool4_output_height_ * pool4_output_width_;
    std::vector<size_t> fc_layers_node_array{fc5_input_node_, fc5_output_node_, fc6_output_node_};
    if (-1 == neural_network_layer5_->Initialize(fc_layers_node_array)) {
        LOG(ERROR) << "Init LeNet Model failed, full connected layer 5 initialize occur error";
        return -1;
    }

    //5. 计算全连接层5的前向计算结果和全连接层6的前向计算结果 这里第4层的输出reshape成3136*1的列向量 送入fc 
    Matrix2d fc5_input_array;
    Matrix::Reshape(max_pooling_layer4_->get_output_array(), fc5_input_node_, 1, fc5_input_array);
    if (-1 == neural_network_layer5_->Predict(fc5_input_array, output_array_)) {
        LOG(ERROR) << "Init LeNet Model failed, full connected layer 5 forward occur error";
        return -1;
    }
    
    return 0;
}

/*
 * LeNet-5 训练  mini-batch 小批的梯度下降
 * 输入 训练样本, 训练标签
 * 一batch的大小
 */
int LeNet::Train(const Matrix4d& training_sample, 
                 const Matrix3d& training_label, 
                 int batch_size) {
    if (batch_size <= 0) {
        LOG(ERROR) << "LeNet-5 train failed, input train batch size <= 0";
        return -1;
    }

    //一epoch的总训练样本大小
    int total_training_sample_size = training_sample.size();
    //一epoch要迭代的次数
    int iterators_count = total_training_sample_size / batch_size;
    for (int i = 0; i < iterators_count; i++) {
        for (int j = 0; j < batch_size; j++) {
            //累加一个batch次反向传播得到的梯度 
            if (-1 == TrainOneSample(training_sample[i * batch_size + j], 
                                     training_label[i * batch_size + j])) {
                LOG(ERROR) << "LeNet-5 train failed";
                return -1;
            }
        }
        //迭代完成一次 更新一次网络权重 得到batch的平均梯度 梯度下降算法优化更新
        UpdateWeights(learning_rate_, batch_size);
    }

    return 0;
}

/*
 * LeNet-5 训练  mini-batch 小批的梯度下降
 * 输入 训练样本, 训练标签
 * 一batch的大小
 */
int LeNet::Train(const ImageMatrix4d& training_sample, 
                 const Matrix3d& training_label, 
                 int batch_size) {
    if (batch_size <= 0) {
        LOG(ERROR) << "LeNet-5 train failed, input train batch size <= 0";
        return -1;
    }

    //一epoch的总训练样本大小
    int total_training_sample_size = training_sample.size();
    //一epoch要迭代的次数
    int iterators_count = total_training_sample_size / batch_size;
    for (int i = 0; i < iterators_count; i++) {
        for (int j = 0; j < batch_size; j++) {
            //累加一个batch次反向传播得到的梯度 
            if (-1 == TrainOneSample(training_sample[i * batch_size + j], 
                                     training_label[i * batch_size + j])) {
                LOG(ERROR) << "LeNet-5 train failed";
                return -1;
            }
        }
        //迭代完成一次 更新一次网络权重 得到batch的平均梯度 梯度下降算法优化更新
        UpdateWeights(learning_rate_, batch_size);

        Matrix2d output_array;
        LOG(INFO) << "迭代完成" << i << "次!";
        Predict(training_sample[8], output_array);
        LOG(INFO) << "loss: " << Loss(output_array, training_label[8]);
    }

    return 0;
}

/*
 * 内部函数 训练一个样本
 * 输入 mnist数据集sample 28*28*1
 * 1. 前向传播 计算输出
 * 2. 反向传播 利用下一层误差项计算上一层误差项 并得到上一层权重梯度
 */
int LeNet::TrainOneSample(const Matrix3d& sample, 
                          const Matrix2d& label) {
    if (-1 == Forward(sample)) {
        LOG(ERROR) << "LeNet-5 train failed, forward occur error";
        return -1;
    }
    if (-1 == Backward(label)) {
        LOG(ERROR) << "LeNet-5 train failed, backward occur error";
        return -1;
    }
    
    return 0;
}

/*
 * 内部函数 训练一个样本
 * 输入 mnist数据集sample 28*28*1
 * 1. 前向传播 计算输出
 * 2. 反向传播 利用下一层误差项计算上一层误差项 并得到上一层权重梯度
 */
int LeNet::TrainOneSample(const ImageMatrix3d& sample, 
                          const Matrix2d& label) {
    if (-1 == Forward(sample)) {
        LOG(ERROR) << "LeNet-5 train failed, forward occur error";
        return -1;
    }
    if (-1 == Backward(label)) {
        LOG(ERROR) << "LeNet-5 train failed, backward occur error";
        return -1;
    }
    
    return 0;
}

/*
 * 更新全连接层5 卷积层3 和卷积层1的权重
 */
void LeNet::UpdateWeights(double learning_rate, int batch_size) {
    neural_network_layer5_->UpdateWeights(learning_rate, batch_size);
    convolutional_layer3_->UpdateWeights(learning_rate, batch_size);
    convolutional_layer1_->UpdateWeights(learning_rate, batch_size);
}

/*
 * 预测 做前向计算 封装forward 得到输出结果
 */
int LeNet::Predict(const Matrix3d& input_array, 
                   Matrix2d& output_array) {
    if (-1 == Forward(input_array)) {
        LOG(ERROR) << "LeNet-5 predict failed, forward occur error";
        return -1;
    }
    output_array = output_array_;

    return 0;
}

/*
 * 预测 做前向计算 封装forward 得到输出结果
 */
int LeNet::Predict(const ImageMatrix3d& input_array, 
                   Matrix2d& output_array) {
    if (-1 == Forward(input_array)) {
        LOG(ERROR) << "LeNet-5 predict failed, forward occur error";
        return -1;
    }
    output_array = output_array_;

    return 0;
}

/*
 * 计算梯度 做反向计算 封装backward 得到输入的误差项
 */
int LeNet::CalcGradient(const Matrix2d& output_array, 
                        Matrix3d& delta_array) {
    if (-1 == Backward(output_array)) {
        LOG(ERROR) << "LeNet-5 Calc Gradient failed, backward occur error";
        return -1;
    }
    delta_array = delta_array_;
    
    return 0;
}

/*
 * LeNet-5的前向计算
 * 一层一层的计算结果 上一层的输出作为下一层的输入
 * 1. 计算卷积层1的前向计算结果 补0填充2 输入28*28*1变32*32*1 filter5*5*32 步长1 得到28*28*32输出特征图
 * 2. 计算池化层2的前向计算结果 输入28*28*32 filter2*2 步长2 下采样降维输出14*14*32
 * 3. 计算卷积层3的前向计算结果 补0填充2 输入14*14*32变18*18*32 filter5*5*64 步长1 得到14*14*64输出特征图
 * 4. 计算池化层4的前向计算结果 输入14*14*64 filter2*2 步长2 下采样降维输出7*7*64
 * 5. 计算全连接层5的前向计算结果 这里第4层的输出reshape成3136*1的列向量 送入fc 得到512*1输出
 * 6. 计算全连接层6的前向计算结果 输入512*1 输出10*1 表示预测的0-9 10个数字的类别结果
 */
int LeNet::Forward(const ImageMatrix3d& input_array) { 
    //1. 计算卷积层1的前向计算结果 补0填充2 输入28*28*1变32*32*1 filter5*5*32 步长1 得到28*28*32输出特征图
    if (-1 == convolutional_layer1_->Forward(input_array)) {
        LOG(ERROR) << "LeNet-5 forward failed, conv layer 1 forward occur error";
        return -1;
    }
    
    //2. 计算池化层2的前向计算结果 输入28*28*32 filter2*2 步长2 下采样降维输出14*14*32
    if (-1 == max_pooling_layer2_->Forward(convolutional_layer1_->get_output_array())) {
        LOG(ERROR) << "LeNet-5 forward failed, pool layer 2 forward occur error";
        return -1;
    }

    //3. 计算卷积层3的前向计算结果 补0填充2 输入14*14*32变18*18*32 filter5*5*64 步长1 得到14*14*64输出特征图
    if (-1 == convolutional_layer3_->Forward(max_pooling_layer2_->get_output_array())) {
        LOG(ERROR) << "LeNet-5 forward failed, conv layer 3 forward occur error";
        return -1;
    }

    //4. 计算池化层4的前向计算结果 输入14*14*64 filter2*2 步长2 下采样降维输出7*7*64
    if (-1 == max_pooling_layer4_->Forward(convolutional_layer3_->get_output_array())) {
        LOG(ERROR) << "LeNet-5 forward failed, pool layer 4 forward occur error";
        return -1;
    } 

    //5. 计算全连接层5的前向计算结果和全连接层6的前向计算结果 这里第4层的输出reshape成3136*1的列向量 送入fc 
    Matrix2d fc5_input_array;
    Matrix::Reshape(max_pooling_layer4_->get_output_array(), fc5_input_node_, 1, fc5_input_array);
    if (-1 == neural_network_layer5_->Predict(fc5_input_array, output_array_)) {
        LOG(ERROR) << "LeNet-5 forward failed, full connected layer 5 forward occur error";
        return -1;
    }
    
    return 0;
}

/*
 * LeNet-5的前向计算
 * 一层一层的计算结果 上一层的输出作为下一层的输入
 * 1. 计算卷积层1的前向计算结果 补0填充2 输入28*28*1变32*32*1 filter5*5*32 步长1 得到28*28*32输出特征图
 * 2. 计算池化层2的前向计算结果 输入28*28*32 filter2*2 步长2 下采样降维输出14*14*32
 * 3. 计算卷积层3的前向计算结果 补0填充2 输入14*14*32变18*18*32 filter5*5*64 步长1 得到14*14*64输出特征图
 * 4. 计算池化层4的前向计算结果 输入14*14*64 filter2*2 步长2 下采样降维输出7*7*64
 * 5. 计算全连接层5的前向计算结果 这里第4层的输出reshape成3136*1的列向量 送入fc 得到512*1输出
 * 6. 计算全连接层6的前向计算结果 输入512*1 输出10*1 表示预测的0-9 10个数字的类别结果
 */
int LeNet::Forward(const Matrix3d& input_array) { 
    //1. 计算卷积层1的前向计算结果 补0填充2 输入28*28*1变32*32*1 filter5*5*32 步长1 得到28*28*32输出特征图
    if (-1 == convolutional_layer1_->Forward(input_array)) {
        LOG(ERROR) << "LeNet-5 forward failed, conv layer 1 forward occur error";
        return -1;
    }

    //2. 计算池化层2的前向计算结果 输入28*28*32 filter2*2 步长2 下采样降维输出14*14*32
    if (-1 == max_pooling_layer2_->Forward(convolutional_layer1_->get_output_array())) {
        LOG(ERROR) << "LeNet-5 forward failed, pool layer 2 forward occur error";
        return -1;
    }

    //3. 计算卷积层3的前向计算结果 补0填充2 输入14*14*32变18*18*32 filter5*5*64 步长1 得到14*14*64输出特征图
    if (-1 == convolutional_layer3_->Forward(max_pooling_layer2_->get_output_array())) {
        LOG(ERROR) << "LeNet-5 forward failed, conv layer 3 forward occur error";
        return -1;
    }

    //4. 计算池化层4的前向计算结果 输入14*14*64 filter2*2 步长2 下采样降维输出7*7*64
    if (-1 == max_pooling_layer4_->Forward(convolutional_layer3_->get_output_array())) {
        LOG(ERROR) << "LeNet-5 forward failed, pool layer 4 forward occur error";
        return -1;
    } 

    //5. 计算全连接层5的前向计算结果和全连接层6的前向计算结果 这里第4层的输出reshape成3136*1的列向量 送入fc 
    Matrix2d fc5_input_array;
    Matrix::Reshape(max_pooling_layer4_->get_output_array(), fc5_input_node_, 1, fc5_input_array);
    if (-1 == neural_network_layer5_->Predict(fc5_input_array, output_array_)) {
        LOG(ERROR) << "LeNet-5 forward failed, full connected layer 5 forward occur error";
        return -1;
    }
    
    return 0;
}

/*
 * LeNet-5的反向计算 从输出层一层层往回计算
 * 先计算输出层误差项 再用此误差项 算出上一层误差项和权重梯度 
 * 1. 计算全连接层5的误差项 和权重梯度 得到最靠输入的fc层误差项数组3136*1 
 * 2. 计算池化层4的误差传递 输入敏感图3136*1 reshape成7*7*64 传递给上一层14*14*64
 * 3. 计算卷积层3的误差项   和权重梯度 输入敏感图14*14*64 传递给上一层14*14*32
 * 4. 计算池化层2的误差传递 输入敏感图14*14*32 传递给上一层28*28*32
 * 5. 计算卷积层1的误差项   和权重梯度 输入敏感图28*28*32 传递给上一层28*28*1
 */
int LeNet::Backward(const Matrix2d& label) {
    //1. 计算全连接层5的误差项 和权重梯度  得到最靠输入的fc层误差项数组3136*1 
    Matrix2d fc5_delta_array;
    Matrix3d pool4_output_delta_array;
    if (-1 == neural_network_layer5_->CalcGradient(output_array_, label, fc5_delta_array)) {
        LOG(ERROR) << "LeNet-5 backward failed, full connected layer 5 backward occur error";
        return -1;
    }
    //将全连接层fc5的误差项数组3136*1 reshape 成池化层4输出的形状7*7*64
    Matrix::Reshape(fc5_delta_array, pool4_output_depth_, pool4_output_height_, 
                    pool4_output_width_, pool4_output_delta_array);
    
    //2. 计算池化层4的误差传递 
    if (-1 == max_pooling_layer4_->Backward(pool4_output_delta_array)) {
        LOG(ERROR) << "LeNet-5 backward failed, pool layer 4 backward occur error";
        return -1;
    }

    //3. 计算卷积层3的误差项 和权重梯度 
    if (-1 == convolutional_layer3_->Backward(max_pooling_layer4_->get_delta_array())) {
        LOG(ERROR) << "LeNet-5 backward failed, conv layer 3 backward occur error";
        return -1;
    }

    //4. 计算池化层2的误差传递
    if (-1 == max_pooling_layer2_->Backward(convolutional_layer3_->get_delta_array())) {
        LOG(ERROR) << "LeNet-5 backward failed, pool layer 2 backward occur error";
        return -1;
    }

    //5. 计算卷积层1的误差项 和权重梯度 
    if (-1 == convolutional_layer1_->Backward(max_pooling_layer2_->get_delta_array())) {
        LOG(ERROR) << "LeNet-5 backward failed, conv layer 1 backward occur error";
        return -1;
    }

    delta_array_ = convolutional_layer1_->get_delta_array();

    return 0;
}

/*
 * 损失函数 求label和预测值的均方误差
 */
double LeNet::Loss(const Matrix2d& output_array, 
                   const Matrix2d& label) const noexcept {
    return Matrix::MeanSquareError(output_array, label);
}



}       //namespace cnn

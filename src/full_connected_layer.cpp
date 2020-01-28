/*
 * =====================================================================================
 *
 *       Filename:  full_connected_layer.cpp
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
#include "full_connected_layer.h"

#include <memory>
#include <vector>
#include <functional>

#include <glog/logging.h>

#include "utility/matrix_math_function.hpp"
#include "utility/matrix_gpu.h"

namespace dnn {

//静态成员的初始化
FullConnectedLayer::SigmoidActivatorCallback FullConnectedLayer::sigmoid_forward_callback_(nullptr);
FullConnectedLayer::SigmoidActivatorCallback FullConnectedLayer::sigmoid_backward_callback_(nullptr);
FullConnectedLayer::ReLuActivatorCallback    FullConnectedLayer::relu_forward_callback_(nullptr);
FullConnectedLayer::ReLuActivatorCallback    FullConnectedLayer::relu_backward_callback_(nullptr);
FullConnectedLayer::Matrix2d FullConnectedLayer::binomial_array_(1, FullConnectedLayer::Matrix1d(1));

FullConnectedLayer::FullConnectedLayer() {
}

FullConnectedLayer::~FullConnectedLayer() {
}

/*
 * 初始化全连接层
 * 初始化权重数组 偏置数组为一个很小的值
 */
int FullConnectedLayer::Initialize(size_t input_node_size, 
                                   size_t output_node_size) {
    input_node_size_ = input_node_size;
    output_node_size_ = output_node_size; 

    //初始化权重数组 和 偏置数组 
    //比如第一层3136个节点 第二层512个 权重就是512行3136列 
    //每行的3136列是权重 和对应神经元输入相乘相加 结果是输出神经元的值
    if (-1 == Random::Uniform(-0.1, 0.1, output_node_size_, input_node_size_, weights_array_)) {
        LOG(ERROR) << "full connected layer initialize failed";
        return -1;
    }
    Matrix::CreateZeros(output_node_size_, 1, biases_array_);
    
    //初始化权重梯度  偏置梯度 输出数组
    Matrix::CreateZeros(output_node_size_, input_node_size, weights_gradient_array_);
    Matrix::CreateZeros(output_node_size_, 1, biases_gradient_array_);
    Matrix::CreateZeros(output_node_size_, 1, output_array_);

    return 0;
}

/*
 * 前向计算 a = f(w .* x + b)  输出等于激活函数(权重数组 点积 输入数组 最后数组和偏置数组相加)
 * 下一层前向计算的输入数组 就是上一层的输出数组
 */  
int FullConnectedLayer::Forward(const Matrix2d& input_array, 
                                bool dropout, float p) {
    //得到本层输入矩阵 也就是本层的节点值
    input_array_ = input_array;
#if GPU
    if (is_input_layer_) {
        if (dropout) {
            if (-1 == calculate::cuda::FullConnectedLayerForward(weights_array_, input_array_, 
                                                                 biases_array_, binomial_array_,
                                                                 output_array_, is_input_layer_,
                                                                 dropout, p)) {
                LOG(ERROR) << "full connected layer forward failed";
                return -1;
            }
        } else {
            if (-1 == calculate::cuda::FullConnectedLayerForward(weights_array_, input_array_, 
                                                                 biases_array_, output_array_,
                                                                 is_input_layer_)) {
                LOG(ERROR) << "full connected layer forward failed";
                return -1;
            }
        }
    } else {
        if (-1 == calculate::cuda::FullConnectedLayerForward(weights_array_, input_array_, 
                                                             biases_array_, output_array_)) {
            LOG(ERROR) << "full connected layer forward failed";
            return -1;
        }
    }
    
#else
    //矩阵相乘  w .* x 得到输出数组
    if (-1 == Matrix::DotProduct(weights_array_, input_array_, output_array_)) {
        LOG(ERROR) << "full connected layer forward failed";
        return -1;
    }
    //矩阵相加 w .* x + b
    if (-1 == Matrix::Add(output_array_, biases_array_, output_array_)) {
        LOG(ERROR) << "full connected layer forward failed";
        return -1;
    }
    
    //激活函数 得到本层输出数组 f(w .* x + b)
    if (is_input_layer_) {
        //输入层就用relu做激活函数 如果是train有dropout 还要dropout一下 测试没有
        if (relu_forward_callback_) {
            //relu_forward_callback_(output_array_, output_array_);
            sigmoid_forward_callback_(output_array_, output_array_);
            if (dropout) {
                if (-1 == Random::DropOut(output_array_, 1, p, 
                                          binomial_array_, 
                                          output_array_)) {
                    LOG(ERROR) << "full connected layer forward failed, dropout occur error";
                    return -1;
                }
            }
        } else {
            LOG(ERROR) << "full connected layer forward failed, relu forward activator is empty";
            return -1;
        }
    } else {
        //输出层用sigmoid激活函数 不用dropout
        if (sigmoid_forward_callback_) {
            sigmoid_forward_callback_(output_array_, output_array_);
        } else {
            LOG(ERROR) << "full connected layer forward failed, sigmoid forward activator is empty";
            return -1;
        }
    }
#endif

    return 0;
}

/*
 * 反向计算 x是本层节点的值 WT是权重数组的转置矩阵 .*是点积 delta_array是下一层的误差数组
 * 本层的误差项 = x * (1 - x) * WT .* delta_array
 * w权重的梯度 就是 delta_array .* xT  下一层的误差项 点积 本层节点值的转置矩阵
 * b偏置的梯度 就是 delta_array
 */
int FullConnectedLayer::Backward(const Matrix2d& output_delta_array, 
                                 bool dropout, float p) {
#if GPU
    if (!is_input_layer_ && dropout) {
        if (-1 == calculate::cuda::FullConnectedLayerBackward(output_delta_array, weights_array_, 
                                                              input_array_, binomial_array_, 
                                                              delta_array_, weights_gradient_array_, 
                                                              biases_gradient_array_, is_input_layer_, 
                                                              dropout, p)) {
            LOG(ERROR) << "full connected layer backward failed";
            return -1;
        }
    } else {
        if (-1 == calculate::cuda::FullConnectedLayerBackward(output_delta_array, weights_array_, 
                                                              input_array_, delta_array_, 
                                                              weights_gradient_array_, 
                                                              biases_gradient_array_)) {
            LOG(ERROR) << "full connected layer backward failed";
            return -1;
        }
    }
#else
    Matrix2d temp_array1;
    if (sigmoid_backward_callback_) {
        // 计算x * (1 - x)
        sigmoid_backward_callback_(input_array_, temp_array1);
    } else {
        LOG(ERROR) << "full connected layer backward failed, sigmoid backward activator is empty";
        return -1;
    }
    
    //计算w的转置矩阵 WT 
    Matrix2d weights_transpose_array;
    if (-1 == Matrix::Transpose(weights_array_, weights_transpose_array)) {
        LOG(ERROR) << "full connected layer backward failed";
        return -1;
    }
    
    Matrix2d temp_array2;
    //计算WT .* delta_array
    if (-1 == Matrix::DotProduct(weights_transpose_array, output_delta_array, temp_array2)) {
        LOG(ERROR) << "full connected layer backward failed";
        return -1;
    }

    //计算x * (1 - x) * WT .* delta_array 得到本层的delta_array
    if (-1 == Matrix::HadamarkProduct(temp_array1, temp_array2, delta_array_)) {
        LOG(ERROR) << "full connected layer backward failed";
        return -1;
    }
    
    //如果有dropout 
    if (!is_input_layer_ && dropout) {
        if (-1 == Matrix::HadamarkProduct(delta_array_, binomial_array_, delta_array_)) {
            LOG(ERROR) << "full connected layer backward failed";
            return -1;
        }
        
        if (-1 == Matrix::MatrixDivValue(delta_array_, 1.0 - p, delta_array_)) {
            LOG(ERROR) << "full connected layer backward failed";
            return -1;
        }
        
        Matrix2d temp_array;
        if (relu_backward_callback_) {
            relu_backward_callback_(input_array_, temp_array);
        }
        if (-1 == Matrix::HadamarkProduct(delta_array_, temp_array, delta_array_)) {
            LOG(ERROR) << "full connected layer backward failed";
            return -1;
        }
    }
    //利用上一层的误差项delta_array 计算weights的梯度 delta_array .* xT
    Matrix2d input_transpose_array;
    Matrix2d weights_gradient_array;
    if (-1 == Matrix::Transpose(input_array_, input_transpose_array)) {
        LOG(ERROR) << "full connected layer backward failed";
        return -1;
    }
    if (-1 == Matrix::DotProduct(output_delta_array, input_transpose_array, weights_gradient_array)) {
        LOG(ERROR) << "full connected layer backward failed";
        return -1;
    }
    
    
    //一个batch反向传播计算的权重梯度累加起来
    Matrix::Add(weights_gradient_array_, weights_gradient_array, weights_gradient_array_);
    //利用上一层的误差项delta_array 计算biases的梯度 delta_array
    //一个batch反向传播计算的偏置梯度累加起来
    Matrix::Add(biases_gradient_array_, output_delta_array, biases_gradient_array_);
#endif    

    return 0;
}

/*
 * 利用梯度下降优化算法(就是让值朝着梯度的反方向走) 更新权重 
 * w = w + learning_rate * w_gradient
 * b = b + learning_rate * b_gradient
 */
void FullConnectedLayer::UpdateWeights(double learning_rate, int batch_size) {
    //梯度下降优化
    Matrix::GradientDescent(weights_gradient_array_, biases_gradient_array_, 
                            learning_rate, batch_size, 
                            weights_array_, biases_array_);
    //将权重梯度置0 方便下个batch计算平均梯度
    Matrix::CreateZeros(Matrix::GetShape(weights_gradient_array_), weights_gradient_array_);
    //将偏置梯度置0 方便下个batch计算平均梯度
    Matrix::CreateZeros(Matrix::GetShape(biases_gradient_array_), biases_gradient_array_);
}

/*
 * 利用梯度下降优化算法(就是让值朝着梯度的反方向走) 更新权重 
 * w = w + learning_rate * w_gradient
 * b = b + learning_rate * b_gradient
 */
void FullConnectedLayer::UpdateWeightsOld(double learning_rate, int batch_size) {
    //得到一个batch的平均权重梯度
    Matrix::MatrixDivValue(weights_gradient_array_, batch_size, weights_gradient_array_);
    //权重的变化数组 
    Matrix2d weights_delta_array;
    Matrix::ValueMulMatrix(learning_rate, weights_gradient_array_, weights_delta_array);
    Matrix::Add(weights_array_, weights_delta_array, weights_array_);
    //将权重梯度置0 方便下个batch计算平均梯度
    Matrix::CreateZeros(Matrix::GetShape(weights_gradient_array_), weights_gradient_array_);

    //得到一个batch的平均偏置梯度
    Matrix::MatrixDivValue(biases_gradient_array_, batch_size, biases_gradient_array_);
    //偏置的变化数组
    Matrix2d biases_delta_array;
    Matrix::ValueMulMatrix(learning_rate, biases_gradient_array_, biases_delta_array);
    Matrix::Add(biases_array_, biases_delta_array, biases_array_);
    //将偏置梯度置0 方便下个batch计算平均梯度
    Matrix::CreateZeros(Matrix::GetShape(biases_gradient_array_), biases_gradient_array_);
}

void FullConnectedLayer::Dump() const noexcept {
    LOG(INFO) << "权重数组:";
    Matrix::MatrixShow(weights_array_); 
    LOG(INFO) << "偏置数组:";
    Matrix::MatrixShow(biases_array_);
}

}       //namespace dnn

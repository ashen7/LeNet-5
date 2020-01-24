/*
 * =====================================================================================
 *
 *       Filename:  filter.cpp
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
#include "filter.h"

#include <vector>

#include <glog/logging.h>

#include "utility/matrix_math_function.hpp"

namespace cnn {

Filter::Filter() {
}

Filter::~Filter() {
}

/*
 * 初始化卷积核
 * 初始化权重数组为一个很小的值 偏置数组为0
 */
void Filter::Initialize(size_t height, size_t width, size_t depth) {
    //生成随机数 来初始化 权重数组 
    //Random::Uniform(-0.0001, 0.0001, depth, height, width, weights_array_);
    Random::Uniform(-0.1, 0.1, depth, height, width, weights_array_);
    bias_ = 0.0;
    
    //初始化梯度为0
    Matrix::CreateZeros(Matrix::GetShape(weights_array_), weights_gradient_array_);
    bias_gradient_ = 0.0;
}

/*
 * 利用梯度下降优化算法(就是让值朝着梯度的反方向走) 更新权重 
 * w = w - learning_rate * w_gradient
 * b = b - learning_rate * b_gradient
 */
void Filter::UpdateWeights(double learning_rate, int batch_size) {
    //得到一个batch里的平均权重梯度 
    Matrix::MatrixDivValue(weights_gradient_array_, batch_size, weights_gradient_array_);
    //权重的变化数组 
    Matrix3d weights_delta_array;
    Matrix::ValueMulMatrix(learning_rate, weights_gradient_array_, weights_delta_array);
    Matrix::Subtract(weights_array_, weights_delta_array, weights_array_);
    //更新完权重后 把权重梯度置0 方便下一个batch 累加它的权重梯度
    Matrix::CreateZeros(Matrix::GetShape(weights_gradient_array_), weights_gradient_array_);

    //得到一个batch里的平均偏置梯度 
    bias_gradient_ = bias_gradient_ / batch_size;
    bias_ -= learning_rate * bias_gradient_;
    //更新完偏置后 把偏置梯度置0 方便下一个batch 累加它的偏置梯度
    bias_gradient_ = 0.0;
}

void Filter::Dump() const noexcept {
    LOG(INFO) << "卷积核权重数组:";
    Matrix::MatrixShow(weights_array_); 
    LOG(INFO) << "偏置: " << bias_;
}

}       //namespace cnn

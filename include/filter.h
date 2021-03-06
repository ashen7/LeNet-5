/*
 * =====================================================================================
 *
 *       Filename:  filter.h
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
#ifndef CNN_FILTER_H_
#define CNN_FILTER_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "utility/matrix_math_function.hpp"

namespace cnn {

//全连接层实现类
class Filter {
public:
    typedef std::vector<std::vector<double>> Matrix2d;
    typedef std::vector<std::vector<std::vector<double>>> Matrix3d;
    typedef std::vector<std::vector<uint8_t>> ImageMatrix2d;
    typedef std::vector<std::vector<std::vector<uint8_t>>> ImageMatrix3d;

    Filter();
    ~Filter();
    Filter(const Filter&) = delete;
    Filter& operator =(const Filter&) = delete;
    Filter(Filter&&) = default;
    Filter& operator =(Filter&&) = default;
    
    void set_weights_array(const Matrix3d& weights_array) noexcept {
        weights_array_ = weights_array;
    }

    //梯度检查时 要小小的改变一下权重 来查看梯度的浮动变化
    Matrix3d& get_weights_array() noexcept {
        return weights_array_;
    }

    void set_bias(double bias) noexcept {
        bias_ = bias;
    }

    double get_bias() const noexcept {
        return bias_;
    }

    void set_weights_gradient_array(const Matrix3d& weights_gradient_array) noexcept {
        //init已初始化0 每次反向传播计算的权重梯度 加起来
        Matrix::Add(weights_gradient_array_, weights_gradient_array, weights_gradient_array_);
    }

    const Matrix3d& get_weights_gradient_array() const noexcept {
        return weights_gradient_array_;
    }

    void set_bias_gradient(double bias_gradient) noexcept {
        //init已初始化0 每次反向传播计算的偏置梯度 加起来
        bias_gradient_ += bias_gradient;
    }

    double get_bias_gradient() const noexcept {
        return bias_gradient_;
    }

public:
    void Initialize(size_t height, size_t width, size_t depth); 
    void UpdateWeights(double learning_rate, int batch_size);
    void Dump() const noexcept;

private:
    Matrix3d weights_array_;          //权重数组(卷积核 共享权重)
    double bias_;                     //偏置
    Matrix3d weights_gradient_array_; //权重梯度数组
    double bias_gradient_;            //偏置梯度
};


}      //namespace cnn

#endif //CNN_FILTER_H_

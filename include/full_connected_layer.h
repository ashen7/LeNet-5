/*
 * =====================================================================================
 *
 *       Filename:  full_connected_layer.h
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
#ifndef DNN_FULL_CONNECTED_LAYER_H_
#define DNN_FULL_CONNECTED_LAYER_H_

#include <memory>
#include <vector>
#include <functional>

namespace dnn {

//全连接层实现类
class FullConnectedLayer {
public:
    typedef std::vector<std::vector<double>> Matrix2d;
    typedef std::vector<std::vector<std::vector<double>>> Matrix3d;
    typedef std::vector<std::vector<uint8_t>> ImageMatrix2d;
    typedef std::vector<std::vector<std::vector<uint8_t>>> ImageMatrix3d;
    typedef std::function<void(const Matrix2d&, Matrix2d&)> ReLuActivatorCallback;

    FullConnectedLayer();
    ~FullConnectedLayer();
    FullConnectedLayer(const FullConnectedLayer&) = delete;
    FullConnectedLayer& operator =(const FullConnectedLayer&) = delete;
    FullConnectedLayer(FullConnectedLayer&&) = default;
    FullConnectedLayer& operator =(FullConnectedLayer&&) = default;
    
    //梯度检查时 要小小的改变一下权重 来查看梯度的浮动变化
    Matrix2d& get_weights_array() noexcept {
        return weights_array_;
    }

    const Matrix2d& get_weights_gradient_array() const noexcept {
        return weights_gradient_array_;
    }

    const Matrix2d& get_output_array() const noexcept {
        return output_array_;
    }

    //优化 复制省略
    const Matrix2d& get_delta_array() const noexcept {
        return delta_array_;
    }

    static void set_relu_forward_callback(ReLuActivatorCallback forward_callback) {
        relu_forward_callback_ = forward_callback;
    }

    static void set_relu_backward_callback(ReLuActivatorCallback backward_callback) {
        relu_backward_callback_ = backward_callback;
    }
    
    static ReLuActivatorCallback get_relu_forward_callback() { 
        return relu_forward_callback_;
    }

    static ReLuActivatorCallback get_relu_backward_callback() {
        return relu_backward_callback_;
    }

public:
    int Initialize(size_t input_node_size, size_t output_node_size); 
    int Forward(const Matrix2d& input_array);
    int Backward(const Matrix2d& output_delta_array);
    void UpdateWeights(double learning_rate, int batch_size);
    void Dump() const noexcept;

protected:
    static ReLuActivatorCallback relu_forward_callback_;  //激活函数的前向计算
    static ReLuActivatorCallback relu_backward_callback_; //激活函数的反向计算 

private:
    bool is_input_layer_;             //是否是输入层
    size_t input_node_size_;          //输入节点
    size_t output_node_size_;         //输出节点
    Matrix2d weights_array_;          //权重数组
    Matrix2d biases_array_;           //偏执数组
    Matrix2d input_array_;            //输入数组
    Matrix2d output_array_;           //输出数组
    Matrix2d delta_array_;            //误差数组
    Matrix2d weights_gradient_array_; //权重梯度数组
    Matrix2d biases_gradient_array_;  //偏置梯度数组
};


}      //namespace dnn

#endif //DNN_FULL_CONNECTED_LAYER_H_

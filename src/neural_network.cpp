/*
 * =====================================================================================
 *
 *       Filename:  neural_network.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2020年01月04日 10时47分21秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "neural_network.h"
#include "full_connected_layer.h"

#include "math.h"

#include <vector>
#include <memory>

#include <glog/logging.h>

#include "utility/matrix_math_function.hpp"
#include "utility/matrix_gpu.h"

namespace dnn {
NeuralNetwork::NeuralNetwork() {
}

NeuralNetwork::~NeuralNetwork() {
    fc_layers_array_.clear();
}

int NeuralNetwork::Initialize(const std::vector<size_t>& fc_layer_nodes_array) {
    //有n层节点 就构造n-1层fc layer
    int fc_layers_array_size = fc_layer_nodes_array.size() - 1;
    fc_layers_array_.reserve(fc_layers_array_size);

    //遍历 初始化全连接层
    for (int i = 0; i < fc_layers_array_size; i++) {
        fc_layers_array_.push_back(std::make_shared<FullConnectedLayer>());
        //初始化
        if (-1 == fc_layers_array_[i]->Initialize(fc_layer_nodes_array[i], 
                                                  fc_layer_nodes_array[i + 1])) {
            LOG(ERROR) << "neural network initialize failed, full connected layer initialize occur error";
            return -1;
        }

        //输入层设置一个flag 用于dropout
        if (0 == i) {
            fc_layers_array_[i]->set_is_input_layer(true);
        } else {
            fc_layers_array_[i]->set_is_input_layer(false);
        }
    }

    //设置全连接层的前向计算激活函数回调
    if (nullptr == FullConnectedLayer::get_sigmoid_forward_callback()) {
#if GPU
        FullConnectedLayer::set_sigmoid_forward_callback(std::bind(Activator::SigmoidForward, 
                                                         std::placeholders::_1,
                                                         std::placeholders::_2));
#else
        FullConnectedLayer::set_sigmoid_forward_callback(std::bind(calculate::cuda::SigmoidForward, 
                                                         std::placeholders::_1,
                                                         std::placeholders::_2));
#endif
    }

    //设置全连接层的反向计算激活函数回调
    if (nullptr == FullConnectedLayer::get_sigmoid_backward_callback()) {
#if GPU
        FullConnectedLayer::set_sigmoid_backward_callback(std::bind(Activator::SigmoidBackward, 
                                                          std::placeholders::_1,
                                                          std::placeholders::_2));
#else
        FullConnectedLayer::set_sigmoid_backward_callback(std::bind(calculate::cuda::SigmoidBackward, 
                                                          std::placeholders::_1,
                                                          std::placeholders::_2));
#endif
    }

    //设置全连接层的前向计算激活函数回调
    if (nullptr == FullConnectedLayer::get_relu_forward_callback()) {
#if GPU
        FullConnectedLayer::set_relu_forward_callback(std::bind(Activator::ReLuForward2d, 
                                                      std::placeholders::_1,
                                                      std::placeholders::_2));
#else
        FullConnectedLayer::set_relu_forward_callback(std::bind(calculate::cuda::ReLuForward2d, 
                                                      std::placeholders::_1,
                                                      std::placeholders::_2));
#endif
    }

    //设置全连接层的反向计算激活函数回调
    if (nullptr == FullConnectedLayer::get_relu_backward_callback()) {
#if GPU
        FullConnectedLayer::set_relu_backward_callback(std::bind(Activator::ReLuBackward2d, 
                                                       std::placeholders::_1,
                                                       std::placeholders::_2));
#else
        FullConnectedLayer::set_relu_forward_callback(std::bind(calculate::cuda::ReLuBackward2d, 
                                                      std::placeholders::_1,
                                                      std::placeholders::_2));
#endif
    }

    return 0;
}

/*
 * 训练网络  
 * 训练集  标签
 * 迭代轮数 和 学习率
 */
int NeuralNetwork::Train(const Matrix3d& training_data_set, 
                         const Matrix3d& labels,
                         int epoch, double learning_rate) {
    //迭代轮数
    for (int i = 0; i < epoch; i++) {
        //遍历每一个输入特征 拿去训练 训练完所有数据集 就是训练完成一轮
        for (int d = 0; d < training_data_set.size(); d++) {
            if (-1 == TrainOneSample(training_data_set[d], labels[d], learning_rate)) {
                LOG(ERROR) << "neural network train failed";
                return -1;
            }
        }
    }

    return 0;
}

/*
 * 内部函数 训练一个样本(输入特征)x 
 * Predict 前向计算 计算网络节点的输出值
 * CalcGradient 反向计算 从输出层开始往前计算每层的误差项 和权重梯度 偏置梯度
 * UpdateWeights 得到了梯度 利用梯度下降优化算法 更新权重和偏置
 */
int NeuralNetwork::TrainOneSample(const Matrix2d& sample, 
                                  const Matrix2d& label, 
                                  double learning_rate) {
    Matrix2d output_array;
    Matrix2d delta_array;
    if (-1 == Predict(sample, output_array)) {
        return -1;
    }

    if (-1 == CalcGradient(output_array, label, delta_array)) {
        return -1;
    }

    //UpdateWeights(learning_rate);

    return 0;
}

/* 
 * 前向计算 实现预测 也就是利用当前网络的权值计算节点的输出值 
 */
int NeuralNetwork::Predict(const Matrix2d& input_array, 
                           Matrix2d& output_array, 
                           bool dropout, float p) {
    for (int i = 0; i < fc_layers_array_.size(); i++) {
        if (0 == i) {
            if (-1 == fc_layers_array_[i]->Forward(input_array, dropout, p)) {
                LOG(ERROR) << "neural network forward failed";
                return -1;
            }
        } else {
            if (-1 == fc_layers_array_[i]->Forward(fc_layers_array_[i - 1]->get_output_array(), dropout, p)) {
                LOG(ERROR) << "neural network forward failed";
                return -1;
            }
        }
        
        if ((i + 1) == fc_layers_array_.size()) {
            output_array = fc_layers_array_[i]->get_output_array();
        }
    }

    return 0;
}


/*
 * 反向计算 计算误差项和梯度 
 * 节点是输出层是 输出节点误差项delta=output(1 - output)(label - output)
 * 通过输出层的delta 从输出层反向计算 依次得到前面每层的误差项 
 * 得到误差项再计算梯度 更新权重使用
 */
int NeuralNetwork::CalcGradient(const Matrix2d& output_array, 
                                const Matrix2d& label, 
                                Matrix2d& fc_input_layer_delta_array, 
                                bool dropout, float p) {
    Matrix2d delta_array; 
    //一个函数 计算之前3个函数的运算
    if (FullConnectedLayer::get_sigmoid_backward_callback()) {
        if (-1 == Matrix::CalcDiff(output_array, label, delta_array)) {
            LOG(ERROR) << "neural network backward failed";
            return -1;
        }
    } else {
        LOG(ERROR) << "neural network backward failed, sigmoid backward activator is empty";
        return -1;
    }

    //从输出层往前反向计算误差项 和梯度
    for (int i = fc_layers_array_.size() - 1; i >= 0; i--) {
        if (i == fc_layers_array_.size() - 1) {
            fc_layers_array_[i]->Backward(delta_array, dropout, p);
        } else {
            //用后一层的delta array 去得到本层的delta array和本层的权重梯度 偏置梯度
            fc_layers_array_[i]->Backward(fc_layers_array_[i + 1]->get_delta_array(), dropout, p);
        }
    }
    
    //第一层fc的delta array赋值出来
    fc_input_layer_delta_array = fc_layers_array_[0]->get_delta_array();

    return 0;
}

//利用梯度下降优化算法 更新网络的权值
/*
 * 反向计算 计算误差项和梯度 
 * 节点是输出层是 输出节点误差项delta=output(1 - output)(label - output)
 * 通过输出层的delta 从输出层反向计算 依次得到前面每层的误差项 
 * 得到误差项再计算梯度 更新权重使用
 */
int NeuralNetwork::CalcGradientOld(const Matrix2d& output_array, 
                                   const Matrix2d& label, 
                                   Matrix2d& fc_input_layer_delta_array, 
                                   bool dropout, float p) {
    //得到output(1 - output)
    Matrix2d delta_array; 
    if (FullConnectedLayer::get_sigmoid_backward_callback()) {
        auto sigmoid_backward_callback = FullConnectedLayer::get_sigmoid_backward_callback();
        sigmoid_backward_callback(output_array, delta_array);
    } else {
        LOG(ERROR) << "neural network backward failed, sigmoid backward activator is empty";
        return -1;
    }
     
    //计算(label - output)
    Matrix2d sub_array; 
    if (-1 == Matrix::Subtract(label, output_array, sub_array)) {
        LOG(ERROR) << "neural network backward failed";
        return -1;
    }

    //再计算output(1 - output)(label - output)  得到输出层的delta array误差项
    if (-1 == Matrix::HadamarkProduct(delta_array, sub_array, delta_array)) {
        LOG(ERROR) << "neural network backward failed";
        return -1;
    }

    //从输出层往前反向计算误差项 和梯度
    for (int i = fc_layers_array_.size() - 1; i >= 0; i--) {
        if (i == fc_layers_array_.size() - 1) {
            fc_layers_array_[i]->Backward(delta_array, dropout, p);
        } else {
            //用后一层的delta array 去得到本层的delta array和本层的权重梯度 偏置梯度
            fc_layers_array_[i]->Backward(fc_layers_array_[i + 1]->get_delta_array(), dropout, p);
        }
    }
    
    //第一层fc的delta array赋值出来
    fc_input_layer_delta_array = fc_layers_array_[0]->get_delta_array();

    return 0;
}

//利用梯度下降优化算法 更新网络的权值
void NeuralNetwork::UpdateWeights(double learning_rate, int batch_size) {
    for (auto fc_layer : fc_layers_array_) {
        fc_layer->UpdateWeights(learning_rate, batch_size);
    }
}

//打印权重数组
void NeuralNetwork::Dump() const noexcept {
    for (auto fc_layer : fc_layers_array_) {
        fc_layer->Dump();
    }
}

//损失函数  计算均方误差 
double NeuralNetwork::Loss(const Matrix2d& output_array, 
                          const Matrix2d& label) const noexcept {
    return Matrix::MeanSquareError(output_array, label);
}

//梯度检查
void NeuralNetwork::GradientCheck(const Matrix2d& sample, 
                                  const Matrix2d& label) {
    //获得网络在当前样本下每个权值的梯度
    Matrix2d output_array;
    Matrix2d delta_array;
    Predict(sample, output_array);
    CalcGradient(output_array, label, delta_array);
    
    double epsilon = 0.0001;
    //遍历全连接层的权重数组 每个值都加减一个很小的值 看loss浮动大不大
    size_t weights_rows = 0;
    size_t weights_cols = 0;
    for (auto fc_layer : fc_layers_array_) {
        auto& weights_array = fc_layer->get_weights_array();
        //元祖解包
        std::tie(weights_rows, weights_cols) = Matrix::GetShape(weights_array);
        for (int i = 0; i < weights_rows; i++) {
            for (int j = 0; j < weights_cols; j++) {
                //依次改变每一个权值 来看看网络的loss情况
                Matrix2d output_array;
                weights_array[i][j] += epsilon;
                Predict(sample, output_array);
                double error_1 = Loss(output_array, label);
                
                weights_array[i][j] -= 2 * epsilon;
                Predict(sample, output_array);
                double error_2 = Loss(output_array, label);
                
                //期待的梯度 e2 - e1 / 2n
                double expect_gradient = (error_2 - error_1) / (2.0 * epsilon);
                //将当前改变的权重还原
                weights_array[i][j] += epsilon;

                LOG(INFO) << "weights{" << i << "}" << "{" << j << "}: "
                          << ", expect - actural: "
                          << expect_gradient << " - "
                          << (fc_layer->get_weights_gradient_array())[i][j];
            }
        }
    }
}

/*
 * 保存模型
 */
int NeuralNetwork::DumpModel(std::shared_ptr<double> weights_array, int& index) {
    for (const auto& fc_layer : fc_layers_array_) {
        if (-1 == Matrix::CopyTo(fc_layer->get_weights_array(), index, weights_array)) {
            LOG(ERROR) << "full connected layer save model failed";
            return -1;
        }
        if (-1 == Matrix::CopyTo(fc_layer->get_biases_array(), index, weights_array)) {
            LOG(ERROR) << "full connected layer save model failed";
            return -1;
        }
    }

    return 0;
}

/*
 * check是否dump成功
 */
bool NeuralNetwork::IsDumpModelSuccess(const Matrix3d& weights,
                                       const Matrix3d& biases) {
    for (int i = 0; i < fc_layers_array_.size(); i++) {
        if (weights[i] != fc_layers_array_[i]->get_weights_array()) {
            LOG(ERROR) << "Dump LeNet-5 Model Failed, full connected layer dump occur error";
            return false;
        }
        if (biases[i] != fc_layers_array_[i]->get_biases_array()) {
            LOG(ERROR) << "Dump LeNet-5 Model Failed, full connected layer dump occur error";
            return false;
        }
    }
    
    return true;
}

}    //namespace dnn


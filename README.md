# LeNet-5
Linux Cpp 手写算法从零实现CNN经典网络LeNet-5
支持cpu/gpu两种运算  
backup目录存放的是预训练的权值文件 
全连接层使用dropout作为正则化 默认设置的神经元丢弃概率为0.5 
每层的超参 和batch size 读取mnist数据集的数目 在conf目录的lenet flagfile修改

运行有三种方式(这里编译的源文件是训练和链接lenet5, 单跑训练或测试方法一样):
    1. python scons 自动化构建脚本
       scons -c && scons -j8 && ./lenet5 -flagfile=conf/lenet5_flagfile_configure

    2. cmake 构建makefile
       cd build && cmake .. && make && mv lenet5 .. && ./lenet5 -flagfile=conf/lenet5_flagfile_configure

    3. bash 脚本编译链接
        ./train_and_evaluate_lenet5.sh
以上方法都可以编译链接运行 需要把项目路径和third party library改成自己的路径
想提前结束运行程序 可以Ctrl + C 会保存当前训练的网络权值到backup目录 释放资源后退出程序
       


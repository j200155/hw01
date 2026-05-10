一、项目简介
本项目基于 PyTorch 实现两种卷积神经网络，在 MNIST 手写数字数据集 上完成训练、测试与可视化对比：
极简 CNN：单卷积层 + 池化 + 全连接层
LeNet-5：经典卷积神经网络（2 卷积 + 2 池化 + 3 全连接）
实验包含数据加载、模型训练、性能评估、结果可视化、模型保存等完整流程。
二、文件结构
hw05/
├── simple_cnn.py        # 极简 CNN 完整代码
├── train_lenet.py       # LeNet-5 完整代码
├── requirements.txt     # 运行环境依赖
├── report.md            # 实验报告（结构、超参、结果对比）
├── debug_notes.md       # 调试记录与问题解决
└── README.md            # 项目说明文档
三、运行环境
Python 版本：3.8 及以上
深度学习框架：PyTorch + TorchVision
系统支持：Windows / macOS / Linux
安装依赖
pip install -r requirements.txt
四、运行方式
1. 运行极简 CNN
   python simple_cnn.py
2. 运行 LeNet-5
   python train_lenet.py
五、功能说明
自动加载数据集：训练集 / 测试集自动划分 + 标准化预处理
模型训练：输出每轮损失、训练准确率
模型测试：在测试集上评估最终准确率与损失
数据可视化：展示数据集样本图片
预测可视化：正确预测标绿色，错误预测标红色
训练曲线保存：保存损失变化图
模型保存：训练完成自动保存最优模型参数
六、输出文件说明
运行后自动生成以下文件：
mnist_samples_*.png：数据集样本展示图
training_loss_*.png：训练损失曲线
predictions_*.png：模型预测结果图
simple_cnn_mnist.pth / lenet5_mnist.pth：训练好的模型文件
七、实验超参数（固定不变）
批量大小：64
训练轮数：5
优化器：Adam
学习率：0.001
损失函数：交叉熵损失（CrossEntropyLoss）
随机种子：42（保证实验可复现）
八、预期结果
极简 CNN：测试准确率 ≈ 98.3%
LeNet-5：测试准确率 ≈ 98.6%
LeNet-5 特征提取能力更强，泛化性能略优于简易 CNN
九、作者信息
作业：hw05
模型：SimpleCNN / LeNet-5
数据集：MNIST
框架：PyTorch

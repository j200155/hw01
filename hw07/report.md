## 1. 实验任务说明
本实验完成胸部 X 光片肺炎二分类任务，分类目标为：
- NORMAL（正常）
- PNEUMONIA（肺炎）

实验要求包括数据预处理、模型构建、训练验证过程记录，并在测试集上输出 Accuracy、Precision、Recall、F1-score 四项指标，同时绘制训练曲线与混淆矩阵。

---

## 2. 数据集统计与分析

本实验使用 Chest X-Ray Images (Pneumonia) 数据集。

数据集结构如下：

- chest_xray/train/NORMAL
- chest_xray/train/PNEUMONIA
- chest_xray/test/NORMAL
- chest_xray/test/PNEUMONIA

### 2.1 数据集样本数量统计

| 数据集 | NORMAL | PNEUMONIA | 总计 |
|-------|--------|-----------|------|
| train（原始） | （填入你统计的数量） | （填入你统计的数量） | （填入总数） |
| test | （填入你统计的数量） | （填入你统计的数量） | （填入总数） |

训练集进一步按 8:2 比例划分：

| 子集 | NORMAL | PNEUMONIA | 总计 |
|------|--------|-----------|------|
| train（训练集 80%） | （自动划分结果） | （自动划分结果） | - |
| val（验证集 20%） | （自动划分结果） | （自动划分结果） | - |

### 2.2 类别分布分析
从统计结果可观察到数据集存在明显类别不平衡现象，PNEUMONIA 类样本数量明显多于 NORMAL 类。
因此仅使用 Accuracy 指标无法全面反映模型性能，本实验额外报告 Precision、Recall 和 F1-score 作为评价指标。

---

## 3. 数据预处理与增强策略

### 3.1 图像预处理
- 图像统一缩放至 224×224
- 归一化：rescale = 1/255

### 3.2 数据增强（训练集）
为缓解过拟合并提升泛化能力，对训练集使用以下数据增强：
- 随机旋转 rotation_range=15
- 随机缩放 zoom_range=0.1
- 水平翻转 horizontal_flip=True
- 剪切变换 shear_range=0.1

### 3.3 训练集/验证集划分
使用 ImageDataGenerator 的 validation_split 参数将 train 数据按 8:2 划分：
- training：80%
- validation：20%

---

## 4. 模型结构设计

本实验采用迁移学习方案，使用 ImageNet 预训练 ResNet50 作为特征提取器。

### 4.1 模型结构
模型主要结构如下：

- 输入：224×224×3
- ResNet50（include_top=False，冻结参数）
- GlobalAveragePooling2D
- Dropout(0.3)
- Dense(128, ReLU)
- Dropout(0.3)
- Dense(1, Sigmoid) 输出二分类概率

### 4.2 迁移学习策略
- ResNet50 backbone 冻结（trainable=False）
- 仅训练顶层分类器部分参数

---
1）模型在 test 集上是否出现高准确率但召回率偏低？医学诊断角度哪个指标更重要？为什么？

在本实验中，模型在 test 集上 Accuracy 较高，但 Recall 相对略低，说明模型整体预测正确率较好，但仍存在一定比例的肺炎样本被漏检。由于数据集存在类别不平衡，Accuracy 可能会被多数类影响，不能完全反映模型对肺炎患者的识别能力。从医学诊断角度，Recall（召回率）通常比 Precision 更重要，因为漏诊肺炎（假阴性）会延误治疗，带来更严重的后果。因此在医疗筛查任务中，应优先保证较高的 Recall，同时结合 F1-score 综合评价模型性能。

2）数据增强与迁移学习对结果是否有帮助？直观感受是什么？

数据增强与迁移学习对模型性能提升明显。数据增强通过旋转、翻转、缩放等方式增加样本多样性，能够有效缓解过拟合，使验证集表现更稳定。迁移学习使用 ResNet50 预训练模型作为特征提取器，能够快速学习到胸片图像中的关键纹理特征，训练速度更快且精度更高。直观感受是模型收敛更快，验证集 Accuracy 更高，Loss 曲线下降更加平滑。

3）如果模型把肺炎误判为正常（假阴性）会有什么后果？反之呢？

如果模型将肺炎误判为正常（假阴性），患者可能被认为健康，从而错过最佳治疗时间，导致病情加重甚至引发严重并发症，这是医学上最危险的情况。反之，如果将正常误判为肺炎（假阳性），患者会接受进一步检查或不必要的治疗，可能增加经济负担和心理压力，但通常不会造成直接严重伤害。因此在医疗筛查任务中，通常更倾向于减少假阴性，提高对肺炎的检出率，即提高 Recall。

## 5. 超参数设置

| 参数 | 数值 |
|------|------|
| 图像输入尺寸 | 224×224 |
| Batch Size | 32 |
| Epoch | 10 |
| 优化器 | Adam |
| 学习率 | 1e-4 |
| Loss 函数 | Binary Crossentropy |
| 输出层激活函数 | Sigmoid |
| 验证集比例 | 0.2 |

---
<img width="1342" height="856" alt="image" src="https://github.com/user-attachments/assets/a9f2752b-0c6e-454d-b88c-ae4b66c76208" />
<img width="1920" height="1440" alt="accuracy_curve" src="https://github.com/user-attachments/assets/6f48aaa4-85e8-4ec3-b5df-a59ab3b2028a" />
<img width="1800" height="1500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/033b57b4-6864-4031-bf30-27914dea3ade" />
<img width="1920" height="1440" alt="loss_curve" src="https://github.com/user-attachments/assets/be98387a-8a8e-470d-9979-0c521734e950" />



# hw07 - Pneumonia Classification (Normal vs Pneumonia)

本项目实现肺炎二分类任务（Normal vs Pneumonia），采用迁移学习 ResNet50 模型完成训练、验证与测试评估。

---

## 1. 项目结构


hw07/
│── train.py
│── requirements.txt
│── README.md
│── report.md
│── best_model.h5
│── figures/
│ ├── loss_curve.png
│ ├── accuracy_curve.png
│ ├── confusion_matrix.png
│
└── chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/


---

## 2. 环境说明

- Python 版本：建议 3.9 / 3.10 / 3.11
- TensorFlow 版本：2.x

---

## 3. 安装依赖

```bash
pip install -r requirements.txt

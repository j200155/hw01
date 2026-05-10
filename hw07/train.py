import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# =======================
# 1. 基本参数
# =======================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "chest_xray"   # 数据集文件夹名称

train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

# 检查路径
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"找不到训练集目录: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"找不到测试集目录: {test_dir}")

print("数据集路径检查通过！")


# =======================
# 2. 统计数据分布（写入报告用）
# =======================
def count_images(folder):
    classes = ["NORMAL", "PNEUMONIA"]
    for cls in classes:
        path = os.path.join(folder, cls)
        num = len(os.listdir(path))
        print(f"{cls}: {num}")

print("\n训练集 train 分布：")
count_images(train_dir)

print("\n测试集 test 分布：")
count_images(test_dir)


# =======================
# 3. 数据增强 + 8:2划分训练/验证
# =======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,      # 8:2划分
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    shear_range=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\n类别索引:", train_generator.class_indices)


# =======================
# 4. 迁移学习 ResNet50 模型
# =======================
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 冻结卷积层

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n模型构建完成！")
model.summary()


# =======================
# 5. 训练模型
# =======================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)


# =======================
# 6. 保存训练曲线
# =======================
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_curve.png", dpi=300)
plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("accuracy_curve.png", dpi=300)
plt.show()

print("\n已保存 loss_curve.png 和 accuracy_curve.png")


# =======================
# 7. 测试集评估
# =======================
test_loss, test_acc = model.evaluate(test_generator)
print("\n========== 测试集评估结果 ==========")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


# =======================
# 8. Precision / Recall / F1 / Confusion Matrix
# =======================
y_true = test_generator.classes

y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

print("\n========== Classification Report ==========")
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NORMAL", "PNEUMONIA"],
            yticklabels=["NORMAL", "PNEUMONIA"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("\n已保存 confusion_matrix.png")


# =======================
# 9. 保存模型
# =======================
model.save("pneumonia_resnet50_model.h5")
print("\n模型已保存为 pneumonia_resnet50_model.h5")

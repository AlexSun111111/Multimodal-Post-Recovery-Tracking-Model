import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.cuda.amp import GradScaler, autocast  # 导入 GradScaler 和 autocast
import os
# 混合精度训练的 GradScaler
from torch.amp import GradScaler, autocast  # 更新为新用法
from sklearn.metrics import accuracy_score, mean_squared_error
from imblearn.over_sampling import SMOTE  # 新增：导入 SMOTE
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 核心数


# 定义列名
personal_columns = ['ID', 'CMR_ID', 'Gender', 'Age']
primary_columns = ['Weight', 'BMI', 'Acute Decompensated Heart Failure', 'NYHA Class', 'Edema Status',
                   'Heart Rate (bpm)', 'Systolic BP', 'Diastolic BP']
illness_columns = ['Hypertension History', 'Coronary Artery Disease', 'Myocardial Infarction', 'Cardiac Stent',
                   'Inflammatory Disease History', 'Kidney Disease', 'Chronic Renal Failure',
                   'Cerebrovascular Disease', 'Smoking History', 'Alcohol History', 'Diabetes Mellitus',
                   'Hyperlipidemia', 'Liver Disease', 'Pulmonary Disease', 'Atrial Fibrillation', 'AV Block',
                   'Pacemaker', 'Other Cardiovascular Disease', 'Other Diseases']

medicine_given_columns = ['medical_indicator']
medical_indicator_columns = ['IVSTd', 'LVPWTd', 'LVDd', 'AoD', 'LAD', 'LVEF', 'BNP', 'TNT', 'CK-MB',
                             'ALT', 'AST',
                             'LDH', 'Total Bilirubin', 'Direct Bilirubin', 'Cholinesterase', 'Total Protein', 'Albumin',
                             'Globulin',
                             'Albumin/Globulin Ratio', 'Glucose', 'Urea', 'Creatinine', 'Uric Acid', 'Triglycerides',
                             'Cholesterol', 'HDL-C', 'LDL-C', 'Apo-A1', 'Apo-B', 'Total Calcium', 'Phosphorus',
                             'Potassium', 'Sodium', 'Chloride', 'CRP', 'eGFR', 'PT', 'INR', 'APTT', 'TT', 'Fibrinogen',
                             'D-dimer',
                             'WBC', 'GRA%', 'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'Basophils',
                             'RBC', 'Hemoglobin', 'Hematocrit', 'Platelets', 'TSH', 'Free T3',
                             'Free T4', 'HbA1c%']
video_columns = ['Myocardial Fibrosis Area']
predict_columns = ['Death', 'Cardiovascular Death', 'Cause of Cardiovascular Death', 'Rehospitalization', 'Cardiology Rehospitalization', 'MACCES Event',
                   'MACCES Cause']

# 数据文件路径
data_file = 'E:/Desktop/IRENE-main/data/text/preprocessed_data_11.26.csv'

# === 加载数据 ===
df = pd.read_csv(data_file)

# 定义列名
binary_columns = ['Death', 'Cardiovascular Death', 'Rehospitalization', 'Cardiology Rehospitalization', 'MACCES Event']
multiclass_columns = ['Cause of Cardiovascular Death', 'MACCES Cause']

# === 数据增强：复制少数类样本 ===
non_zero_indices = np.any(df[binary_columns].values != 0, axis=1)  # 查找每一行中至少有一个标签不为0的行
df_non_zero = df[non_zero_indices]

# 计算需要复制的样本
num_needed_samples = len(df) - len(df_non_zero)  # 需要增加的样本数量

# 如果需要的样本数量大于 0，就进行复制
if num_needed_samples > 0:
    # 计算复制的次数，保证目标样本数不超过
    num_repeats = num_needed_samples // len(df_non_zero)  # 整数除法得到重复的次数
    remainder = num_needed_samples % len(df_non_zero)  # 余数表示额外需要的样本

    # 首先复制足够的样本
    df = pd.concat([df, df_non_zero.sample(n=len(df_non_zero) * num_repeats, replace=True)], ignore_index=True)

    # 然后复制余下的样本
    df = pd.concat([df, df_non_zero.sample(n=remainder, replace=True)], ignore_index=True)

    print(f"Original dataset size: {len(df)}")
else:
    print("No replication needed. Dataset is already balanced.")

# === 提取数值特征 ===
# 使用中位数填充数值特征中的缺失值
imputer = SimpleImputer(strategy='median')
numerical_columns = [col for col in df.columns if col not in ['medical_indicator', 'ID', 'CMR_ID'] + binary_columns + multiclass_columns]
numerical_features = imputer.fit_transform(df[numerical_columns].values.astype(np.float32))

# 归一化
scaler = MinMaxScaler()
numerical_features = scaler.fit_transform(numerical_features)

# === 提取标签 ===
binary_labels = df[binary_columns].values.astype(np.float32)
multiclass_labels = df[multiclass_columns].values.astype(np.int64)

# 转换为张量
numerical_tensor = torch.tensor(numerical_features, dtype=torch.float32)
binary_tensor = torch.tensor(binary_labels, dtype=torch.float32)
multiclass_tensor = torch.tensor(multiclass_labels, dtype=torch.long)

# === 加载 BERT 分词器 ===
texts = df['medical_indicator'].values
tokenizer = BertTokenizer.from_pretrained('./bert_localpath/')
inputs = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# === 创建数据集 ===
dataset = TensorDataset(input_ids, attention_mask, numerical_tensor, binary_tensor, multiclass_tensor)

# 划分训练集和验证集（80/20 划分）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 打印数据集大小
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

import numpy as np
import pandas as pd
# === 定义多模态模型 ===
class MultiModalModel(nn.Module):
    def __init__(self, config, num_numerical_features, num_binary, num_multiclass):
        super(MultiModalModel, self).__init__()

        # BERT部分
        self.bert = BertModel(config)
        self.fc_text = nn.Linear(config.hidden_size, 256)

        # 数值特征部分
        self.fc_num = nn.Sequential(
            nn.Linear(num_numerical_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )

        # 融合文本和数值特征的部分
        self.fc_fusion = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )

        # 添加残差连接层
        self.resnet_fusion_fc1 = nn.Linear(256, 256)
        self.resnet_fusion_fc2 = nn.Linear(256, 256)
        self.resnet_fusion_relu = nn.ReLU()
        self.resnet_fusion_dropout = nn.Dropout(0.2)

        # 二分类任务
        self.fc_binary = nn.Linear(256, num_binary)

        # 多分类任务
        self.fc_multiclass = nn.ModuleList([nn.Linear(256, n) for n in num_multiclass])

        # 添加最终的残差连接层
        self.resnet_final_fc1 = nn.Linear(256, 256)
        self.resnet_final_fc2 = nn.Linear(256, 256)

    def forward(self, input_ids, attention_mask, numerical_features):
        # BERT 输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.fc_text(bert_output.pooler_output)

        # 数值特征部分
        num_features = self.fc_num(numerical_features)

        # 融合文本和数值特征
        combined_features = torch.cat([text_features, num_features], dim=1)
        fusion_out = self.fc_fusion(combined_features)

        # 残差连接部分 1
        residual = fusion_out
        out = self.resnet_fusion_fc1(fusion_out)
        out = self.resnet_fusion_relu(out)
        out = self.resnet_fusion_dropout(out)
        out = self.resnet_fusion_fc2(out)
        fusion_out = out + residual  # 残差连接

        # 残差连接部分 2（进一步融合）
        residual_final = fusion_out
        out = self.resnet_final_fc1(fusion_out)
        out = self.resnet_fusion_relu(out)
        out = self.resnet_fusion_fc2(out)
        fusion_out = out + residual_final  # 残差连接

        # 输出二分类和多分类结果
        binary_logits = self.fc_binary(fusion_out)
        multiclass_logits = [fc(fusion_out) for fc in self.fc_multiclass]

        return binary_logits, multiclass_logits


# 初始化模型
config = BertConfig.from_pretrained('./bert_localpath/config.json')
model = MultiModalModel(
    config=config,
    num_numerical_features=numerical_tensor.shape[1],
    num_binary=len(binary_columns),
    num_multiclass=[df[col].nunique() for col in multiclass_columns]
).cuda()

# === 损失函数与优化器 ===
# 加权 BCE 损失函数，用于处理数据不平衡问题
positive_weights = torch.tensor([len(binary_labels) / (2.0 * binary_labels[:, i].sum()) for i in range(len(binary_columns))]).cuda()
bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
ce_loss_fn = nn.CrossEntropyLoss()  # 多分类损失
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# **新增：动态学习率调度器**
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# === 强化学习奖励机制 ===
def compute_reward(true_labels, predicted_labels, task_weights=None):
    """
    计算奖励函数，强化对特定任务（如 Death 和 Cardiovascular Death）的关注。
    :param true_labels: 真实标签 (n_samples, n_tasks)
    :param predicted_labels: 模型预测 (n_samples, n_tasks)
    :param task_weights: 每个任务的奖励权重 (list)，默认为均等
    :return: 奖励值 (Tensor)
    """
    # 默认任务权重：如果未提供 task_weights，则每个任务的权重均为 1.0
    if task_weights is None:
        task_weights = [1.0] * true_labels.shape[1]
    rewards = np.zeros_like(true_labels, dtype=float)  # 初始化奖励矩阵
    for task_idx in range(true_labels.shape[1]):
        task_reward = (true_labels[:, task_idx] == predicted_labels[:, task_idx]).astype(float)
        task_reward[true_labels[:, task_idx] == 1] *= 2.0  # 对少数类（正类）增加奖励
        task_reward *= task_weights[task_idx]  # 按任务权重调整奖励
        rewards[:, task_idx] = task_reward

    return torch.tensor(rewards, dtype=torch.float32).cuda()

# === 早停机制 ===
# **新增：早停机制实现**
class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

early_stopping = EarlyStopping(patience=10, verbose=True)
"""
def apply_logical_constraints(binary_preds, multiclass_preds):
    # 确保输入是 NumPy 数组
    binary_preds = np.array(binary_preds)
    multiclass_preds = np.array(multiclass_preds)

    # 遍历每个批次（batch）
    for batch_idx in range(binary_preds.shape[0]):
        # 遍历每个样本（sample）在当前批次中
        for sample_idx in range(binary_preds.shape[1]):
            # 1. Rehospitalization depends on Cardiology Rehospitalization
            if binary_preds[batch_idx, sample_idx, 2] == 0:  # 如果 Rehospitalization == 0
                binary_preds[batch_idx, sample_idx, 3] = 0  # 强制 Cardiology Rehospitalization == 0
            elif binary_preds[batch_idx, sample_idx, 3] == 1:  # 如果 Cardiology Rehospitalization == 1
                binary_preds[batch_idx, sample_idx, 2] = 1  # 强制 Rehospitalization == 1

            # 2. Death depends on Cardiovascular Death
            if binary_preds[batch_idx, sample_idx, 0] == 0:  # 如果 Death == 0
                binary_preds[batch_idx, sample_idx, 1] = 0  # 强制 Cardiovascular Death == 0
                multiclass_preds[batch_idx, sample_idx, 0] = 0  # 无需预测 Cause of Cardiovascular Death
            elif binary_preds[batch_idx, sample_idx, 1] == 1:  # 如果 Cardiovascular Death == 1
                # 保证 Cause of Cardiovascular Death 为一个有效类别
                if multiclass_preds[batch_idx, sample_idx, 0] == 0:
                    # 选择最大概率的类别
                    multiclass_preds[batch_idx, sample_idx, 0] = np.argmax(multiclass_preds[batch_idx, sample_idx])  # 选择最有可能的类别

            # 3. MACCES Cause depends on MACCES Event
            if binary_preds[batch_idx, sample_idx, 4] == 0:  # 如果 MACCES Event == 0
                multiclass_preds[batch_idx, sample_idx, 1] = 0  # 无需预测 MACCES Cause
            elif binary_preds[batch_idx, sample_idx, 4] == 1:  # 如果 MACCES Event == 1
                # 保证 MACCES Cause 为一个有效类别
                if multiclass_preds[batch_idx, sample_idx, 1] == 0:
                    multiclass_preds[batch_idx, sample_idx, 1] = np.argmax(multiclass_preds[batch_idx, sample_idx])  # 选择最有可能的类别

    return binary_preds, multiclass_preds
"""


# 权重保存路径
save_dir = "Train/weights"
os.makedirs(save_dir, exist_ok=True)
best_weights_path = os.path.join(save_dir, "best_weights.pth")
latest_weights_path = os.path.join(save_dir, "latest_weights.pth")
predictions_dir = "Train/predictions/v4_1212_v1"
os.makedirs(predictions_dir, exist_ok=True)

# 训练与验证
# 初始化损失列表
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')
epochs = 200

for epoch in range(epochs):
    # === 训练 ===
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch_input_ids, batch_attention_mask, numerical_data, binary_labels, multiclass_labels = (
            batch[0].cuda(),
            batch[1].cuda(),
            batch[2].cuda(),
            batch[3].cuda(),
            batch[4].cuda()
        )

        optimizer.zero_grad()
        binary_logits, multiclass_logits = model(batch_input_ids, batch_attention_mask, numerical_data)

        # 二分类损失
        binary_loss = bce_loss_fn(binary_logits, binary_labels)

        # 多分类损失
        multiclass_loss = sum(
            ce_loss_fn(multiclass_logits[i], multiclass_labels[:, i]) for i in range(len(multiclass_logits)))

        # 计算强化学习奖励
        binary_predictions = (torch.sigmoid(binary_logits) > 0.5).float()
        rewards = compute_reward(binary_labels.cpu().numpy(), binary_predictions.cpu().numpy(),
                                 task_weights=[5.0, 1.0, 5.0, 1.0, 1.0])
        rl_loss = -torch.mean(rewards * torch.log(binary_predictions + 1e-8))  # 强化学习损失

        # 总损失
        loss = binary_loss + multiclass_loss + rl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # 添加训练损失到列表
    print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")

    # === 验证部分 ===
    model.eval()
    val_loss = 0.0
    val_binary_preds, val_binary_labels = [], []
    val_multiclass_preds, val_multiclass_labels = [], []

    # 保存损失和准确率
    binary_task_acc = []
    multiclass_task_acc = []

    with torch.no_grad():
        for batch in val_loader:
            batch_input_ids, batch_attention_mask, numerical_data, binary_labels, multiclass_labels = (
                batch[0].cuda(),
                batch[1].cuda(),
                batch[2].cuda(),
                batch[3].cuda(),
                batch[4].cuda()
            )

            # 前向传播
            binary_logits, multiclass_logits = model(batch_input_ids, batch_attention_mask, numerical_data)

            # 计算损失
            binary_loss = bce_loss_fn(binary_logits, binary_labels)  # 二分类任务损失
            multiclass_loss = sum(
                ce_loss_fn(multiclass_logits[i], multiclass_labels[:, i]) for i in range(len(multiclass_logits))
            )  # 多分类任务损失
            loss = binary_loss + multiclass_loss  # 总验证损失
            val_loss += loss.item()

            # 保存二分类任务预测结果
            binary_preds = (torch.sigmoid(binary_logits) > 0.5).int().cpu().numpy()  # 阈值 0.5 判定为正类
            val_binary_preds.extend(binary_preds.tolist())  # 将预测结果转为列表并添加到 val_binary_preds
            val_binary_labels.extend(binary_labels.cpu().numpy().tolist())  # 将真实标签转为列表并添加到 val_binary_labels

            # 保存多分类任务预测结果
            multiclass_preds = np.stack([torch.argmax(logit, dim=1).cpu().numpy() for logit in multiclass_logits],
                                        axis=1)
            val_multiclass_preds.extend(multiclass_preds.tolist())  # 转为列表并添加
            val_multiclass_labels.extend(multiclass_labels.cpu().numpy().tolist())  # 转为列表并添加

    # 计算平均验证损失
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  # 添加验证损失到列表
    # 调整学习率
    scheduler.step(val_loss)
    # 保存模型权重
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_weights_path)
    # 早停机制
    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

    # 转换为 NumPy 数组
    val_binary_preds = np.vstack(val_binary_preds)
    val_binary_labels = np.vstack(val_binary_labels)
    val_multiclass_preds = np.vstack(val_multiclass_preds)
    val_multiclass_labels = np.vstack(val_multiclass_labels)

    # 打印平均验证损失
    print(f"Epoch [{epoch + 1}/{epochs}] - Validation Loss: {avg_val_loss:.4f}")

    # 计算每个任务的准确率
    print("\n=== Accuracy Results ===")
    print("Binary Tasks:")
    for i, col in enumerate(binary_columns):
        acc_binary = accuracy_score(val_binary_labels[:, i], val_binary_preds[:, i])  # 二分类任务准确率
        binary_task_acc.append(acc_binary)
        print(f"  Task: {col} - Accuracy: {acc_binary:.4f}")

    print("Multiclass Tasks:")
    for i, col in enumerate(multiclass_columns):
        acc_multiple = accuracy_score(val_multiclass_labels[:, i], val_multiclass_preds[:, i])  # 多分类任务准确率
        multiclass_task_acc.append(acc_multiple)
        print(f"  Task: {col} - Accuracy: {acc_multiple:.4f}")
    # 计算总体准确率
    # 将所有二分类和多分类的预测结果和标签结合起来，计算总的准确率
    # 将数组展平（flatten），以便进行总体准确率计算
    flat_binary_preds = val_binary_preds.flatten()
    flat_binary_labels = val_binary_labels.flatten()

    flat_multiclass_preds = val_multiclass_preds.flatten()
    flat_multiclass_labels = val_multiclass_labels.flatten()

    # 合并所有二分类和多分类的预测和标签，计算总的准确率
    total_preds = np.concatenate([flat_binary_preds, flat_multiclass_preds])
    total_labels = np.concatenate([flat_binary_labels, flat_multiclass_labels])

    # 计算总体准确率
    total_accuracy = accuracy_score(total_labels, total_preds)
    val_accuracies.append(total_accuracy)
    # 输出当前轮次的验证结果
    print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {total_accuracy:.4f}")
    # 保存预测结果到文本文件（每隔 10 轮保存）
    if (epoch + 1) % 10 == 0:
        predictions_file = os.path.join(predictions_dir, f"predictions_epoch_{epoch + 1}.txt")
        with open(predictions_file, "w") as f:
            f.write(f"Epoch [{epoch + 1}/{epochs}] - Validation Loss: {avg_val_loss:.4f}\n\n")
            f.write("=== Binary Task Predictions ===\n")
            for i, col in enumerate(binary_columns):
                f.write(f"Task: {col}\n")
                f.write(f"Predictions: {val_binary_preds[:, i]}\n")
                f.write(f"True Labels: {val_binary_labels[:, i]}\n\n")

            f.write("=== Multiclass Task Predictions ===\n")
            for i, col in enumerate(multiclass_columns):
                f.write(f"Task: {col}\n")
                f.write(f"Predictions: {val_multiclass_preds[:, i]}\n")
                f.write(f"True Labels: {val_multiclass_labels[:, i]}\n\n")
        print(f"Predictions saved to {predictions_file}")

# 保存验证集每一轮的损失和准确率到 DataFrame
val_df = pd.DataFrame({
    'Epoch': range(1, epochs + 1),
    'Validation Loss': val_losses,
    'Validation Accuracy': val_accuracies
})

# 保存到 Excel 或 CSV 文件
val_df.to_excel('validation_results.xlsx', index=False)  # 保存为 Excel 文件
# val_df.to_csv('validation_results.csv', index=False)  # 如果你想保存为 CSV 文件

print(f"验证集的损失和准确率已保存到 'validation_results.xlsx'")


# 绘制损失曲线图
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

"""
MACCES具体原因:
1心衰相关，包含：心衰加重、心衰恶化、心衰入院、心衰加重住院等。
2心律失常相关，包含：房颤复发、射频消融等。
3脑血管相关，包含：脑梗、黑蒙、晕倒等。
4感染/炎症相关，包含：心肌炎等。
5其他原因，未明确归类或不常见的原因。

心源性死亡具体原因
1心衰相关：如“心衰加重”、“心衰恶化”、“心衰，肺部异物”等。
2心肌梗塞相关：如“致命心梗”。
3感染相关：如“新冠心肌炎”。
4其他原因：暂未分类的其他原因
"""
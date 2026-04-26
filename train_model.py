import os
import torch
import pandas as pd
from torch import nn
import torch.optim as optim
from models.Diff_all import DiffUNet  # 导入的模型
from torch.utils.data import DataLoader
from scripts.utils import get_input_image_names
from sklearn.model_selection import train_test_split
from scripts.data_preprocessing import TrainDataset, ValDataset


GLOBAL_PATH = '95-Cloud dataset'
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, 'Training')
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'visualization')

# 参数值
img_rows, img_cols = 192, 192  # 目标图像尺寸
num_of_channels = 4
num_of_classes = 1
starting_learning_rate = 1e-5
end_learning_rate = 1e-8
max_num_epochs = 100
val_ratio = 0.2
patience = 15
decay_factor = 0.7
batch_sz = 12
num_worker = 8
experiment_name = "Cloud-Detection"
weights_path = 'outputs/weight/Diff-all_U-Net'  # 模型训练最优参数的保存路径
train_resume = False  # 是否启用断点续训（True:启用；False:不启用）
resume_path = "outputs/checkpoints/Diff-all_U-Net/model_epoch_.pth"  # 指定恢复的检查点路径

train_patches_csv_name = 'training_patches_95-cloud_nonempty.csv'
df_train_img = pd.read_csv(os.path.join(TRAIN_FOLDER, train_patches_csv_name))
train_img, train_msk = get_input_image_names(df_train_img, TRAIN_FOLDER, if_train=True)

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = DiffUNet()  # 训练使用的模型
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # 损失函数
    min_val_loss = float('inf')  # 初始化最小验证损失为正无穷大
    optimizer = optim.Adam(model.parameters(), lr=starting_learning_rate)  # 优化器
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=decay_factor, cooldown=0,
                                                     patience=patience, min_lr=end_learning_rate, verbose=True)
    # 数据分割（将训练数据和标签分割为训练集和验证集）
    train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(train_img, train_msk,
                                                                                      test_size=val_ratio,
                                                                                      random_state=42, shuffle=True)
    # 数据加载：加载预处理好的训练和验证集
    train_dataset = TrainDataset(list(zip(train_img_split, train_msk_split)), img_rows, img_cols)
    val_dataset = ValDataset(list(zip(val_img_split, val_msk_split)), img_rows, img_cols)
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_worker)

    # 是否重新开始训练
    # 保存检查点（每个epoch结束时保存）,明确分离学习率和调度器
    def save_checkpoint(epoch_resume, model_resume, optimizer_resume, loss_resume,
                        lr_resume, scheduler_resume=None, save_dir="outputs/checkpoints/Diff-all_U-Net"):
        # 自动创建目录
        os.makedirs(save_dir, exist_ok=True)
        # 保存完整训练状态
        checkpoint = {
            "epoch": epoch_resume,
            "model_state_dict": model_resume.state_dict(),
            "optimizer_state_dict": optimizer_resume.state_dict(),
            "loss": loss_resume,
            "lr": lr_resume,  # 保存当前学习率数值
            "scheduler_state_dict": scheduler_resume.state_dict() if scheduler_resume else None  # 保存调度器状态
            # 可扩展保存其他参数（如学习率调度器状态）
        }
        torch.save(checkpoint, os.path.join(save_dir, f"model_epoch_{epoch_resume + 1}.pth"))
        print(f"已保存第 {epoch_resume + 1} 轮的模型参数至：{save_dir}")

    # 加载检查点
    def load_checkpoint(resume_path, model_resume, optimizer_resume, scheduler_resume=None):
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path)
            model_resume.load_state_dict(checkpoint["model_state_dict"])
            optimizer_resume.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_resume.param_groups[0]['lr'] = checkpoint["lr"]  # 设置优化器学习率（关键修复点）
            return checkpoint["epoch"] + 1  # 返回下一轮起始epoch
        else:
            print(f"Warning: Checkpoint {resume_path} not found, starting from scratch")
            return 0

    # 训练循环
    start_epoch = 0  # 默认起始epoch
    # 根据控制参数决定是否恢复训练
    if train_resume:
        start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"断点续训 epoch： {start_epoch + 1}")
        print(f"恢复后的学习率:   {optimizer.param_groups[0]['lr']}")  # 验证学习率恢复
        print(f"实验名称：       {experiment_name}")
        print(f"图像尺寸：       {img_rows} * {img_cols}")
        print(f"通道数：         {num_of_channels}")
        print(f"批次大小：       {batch_sz}")
    else:
        print("重新训练")
        print(f"实验名称：       {experiment_name}")
        print(f"图像尺寸：       {img_rows} * {img_cols}")
        print(f"通道数：         {num_of_channels}")
        print(f"开始学习率：      {starting_learning_rate}")
        print(f"批次大小：       {batch_sz}")

    # 保存训练日志
    with open('outputs/logs/Diff-all_U-Net_training_log.txt', 'w') as log_file:
        total_batches = len(train_loader)  # 直接获取批次数
        log_file.write(f'Total batches: {total_batches}\n')  # 写入总批次

        # 训练过程
        for epoch in range(start_epoch, max_num_epochs):
            model.train()
            running_loss = 0.0  # 记录当前epoch的总损失。
            total_batches = len(train_loader)  # 计算总批次数量，整除批次大小
            # 训练集训练
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()  # 清空梯度
                outputs = model(inputs)  # 向前传播
                loss = criterion(outputs, targets)
                # 计算神经网络参数梯度的函数，计算损失函数相对于模型参数的梯度
                loss.backward()  # 反向传播
                optimizer.step()  # 使用优化器（如 SGD、Adam 等）来更新模型的参数
                running_loss += loss.item() # 记录损失
                # 打印当前批次的信息并写入文件
                if batch_idx % 100 == 0:  # 每100个批次打印一次
                    log_message = (f"Epoch [{epoch + 1}/{max_num_epochs}], "
                                   f"Batch [{batch_idx + 1}/{total_batches}], "
                                   f"Loss: {loss.item():.4f}\n")
                    print(log_message.strip())  # 控制台打印
                    log_file.write(log_message)  # 写入文件
            # 验证模式
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            # 平均训练、验证损失
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            # 写入每个轮次的训练和验证损失
            log_file.write(f"Epoch {epoch + 1}/{max_num_epochs}, "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}\n")
            # 根据该轮平均loss值更新学习率
            scheduler.step(avg_val_loss)
            # 跟踪打印学习率
            current_lr = optimizer.param_groups[0]['lr']
            # 使用Python的f-string格式化字符串来打印当前的学习率、损失和训练轮次
            log_file.write(f"Epoch {epoch + 1}/{max_num_epochs}, "
                           f"Learning Rate: {current_lr:.6f}\n")

            # 保存训练完成后最优的模型参数
            if avg_val_loss < min_val_loss:
                # 保存该轮次平均验证loss值
                min_val_loss = avg_val_loss
                # 自动创建最优参数保存目录
                os.makedirs(weights_path, exist_ok=True)
                # 文件保存路径
                file_name_best = f"Diff-all_U-Net_model_best.pth"
                save_path_best = os.path.join(weights_path, file_name_best)
                # 模型参数保存
                torch.save(model.state_dict(), save_path_best)
                print(f"已保存第 {epoch + 1} 轮的模型参数至：{save_path_best}")

            # 模型检查点
            save_checkpoint(epoch, model, optimizer, loss, current_lr)

            # 是否停止训练
            if current_lr < end_learning_rate:
                print("学习率低于0.00000001，停止训练")
                break


if __name__ == '__main__':
    train()

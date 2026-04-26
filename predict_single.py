import os
import torch
import numpy as np
import pandas as pd
import tifffile as tiff
from models.SE_model import UNet
from torch.utils.data import DataLoader
from scripts.data_preprocessing import PredictionDataset
from scripts.utils import get_input_image_names


GLOBAL_PATH = '38-Cloud dataset'
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'visualization')
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, 'Training')
PRED_FOLDER = os.path.join(GLOBAL_PATH, 'Predictions')

# 重构后的图像大小
img_rows = 384
img_cols = 384
num_of_channels = 4
num_of_classes = 1
batch_sz = 1
max_bit = 65535  # 最大灰度值
experiment_name = "Cloud-Detection"
weights_path = os.path.join(GLOBAL_PATH, experiment_name + '.pth')

# 获取输入图像名称
predict_patches_csv_name = 'predict_patches_38-Cloud.csv'
df_predict_img = pd.read_csv(os.path.join(TRAIN_FOLDER, predict_patches_csv_name))
predict_img, predict_ids = get_input_image_names(df_predict_img, TRAIN_FOLDER, if_train=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prediction_single():
    model = UNet(num_of_channels=num_of_channels,
                 num_of_classes=num_of_classes)
    model.to(device)
    # 加载模型和权重
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # 数据加载
    predict_dataset = PredictionDataset(predict_img, img_rows, img_cols, max_bit)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_sz, shuffle=False)

    # 预测
    pred_dir = experiment_name + '_outputs_of_prediction'
    os.makedirs(os.path.join(PRED_FOLDER, pred_dir), exist_ok=True)

    with torch.no_grad():
        for images, image_ids in zip(predict_loader, predict_ids):
            images = images.to(device)  # 将数据移动到 GPU 上
            outputs = model(images)
            outputs = outputs.cpu().numpy()  # 将数据移回 CPU 并转换为 numpy 数组
            image = outputs[0, 0, :, :]  # 取出第一个通道的预测结果
            tiff.imwrite(os.path.join(PRED_FOLDER, pred_dir, str(image_ids)), image.astype(np.float32))

    print("Prediction completed and saved.")


if __name__ == "__main__":
    prediction_single()  # 调用 prediction

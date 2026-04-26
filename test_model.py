import os
import torch
import numpy as np
import pandas as pd
import tifffile as tiff
from models.ViT_UNet import UNetViT
from scripts.utils import get_input_image_names
from torch.utils.data import DataLoader
from scripts.data_preprocessing import PredictionDataset


GLOBAL_PATH = '95-Cloud dataset'
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'Test')
PRED_FOLDER = os.path.join(GLOBAL_PATH, 'Predictions')
WEIGHT_PATH = 'outputs/weight/ViT_U-Net'  # 模型预测加载权重路径

img_rows = 192
img_cols = 192
num_of_channels = 4
num_of_classes = 1
batch_sz = 1
num_workers = 4
weight_name = "ViT_U-Net_model_best.pth"
experiment_name = 'ViT_U-Net'  # 本次实验使用的模型
weights_path = os.path.join(WEIGHT_PATH, weight_name)

# getting input images names
test_patches_csv_name = 'test_patches_38-Cloud.csv'
df_test_img = pd.read_csv(os.path.join(TEST_FOLDER, test_patches_csv_name))
test_img, test_ids = get_input_image_names(df_test_img, TEST_FOLDER, if_train=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prediction():
    model = UNetViT()
    model.to(device)
    # 加载模型和权重
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # 数据加载
    test_dataset = PredictionDataset(test_img, img_rows, img_cols)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz,
                             num_workers=num_workers, shuffle=False)

    # 预测
    pred_dir = experiment_name + '_train_96_test_192'
    pred_file = 'ViT_U-Net_best'
    os.makedirs(os.path.join(PRED_FOLDER, pred_dir), exist_ok=True)
    pred = os.path.join(PRED_FOLDER, pred_dir)
    os.makedirs(os.path.join(pred, pred_file), exist_ok=True)

    with torch.no_grad():
        for images, image_ids in zip(test_loader, test_ids):
            images = images.to(device)  # 将数据移动到 GPU 上
            outputs = model(images)
            outputs = outputs.cpu().numpy()  # 将数据移回 CPU 并转换为 numpy 数组
            image = outputs[0, 0, :, :]  # 取出第一个通道的预测结果
            tiff.imwrite(os.path.join(pred, pred_file, str(image_ids)), image.astype(np.float32))

    print("Prediction completed and saved.")


if __name__ == "__main__":
    prediction()  # 调用 prediction

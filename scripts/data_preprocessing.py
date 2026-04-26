import torch
import random
import numpy as np
from skimage.exposure import rescale_intensity
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.transform import resize
from scripts.data_augmentation import flip_horizontal, flip_vertical, rotate_image, adjust_brightness
from scripts.data_augmentation import add_gaussian_noise, apply_blur, apply_occlusion


# 自定义训练数据集加载器类
class TrainDataset(Dataset):
    def __init__(self, zip_list, img_rows, img_cols):
        self.zip_list = zip_list
        self.img_rows = img_rows
        self.img_cols = img_cols

    def __len__(self):
        return len(self.zip_list)

    def __getitem__(self, idx):
        file, mask = self.zip_list[idx]

        # 读取四通道图像
        image_red = imread(file[0])
        image_green = imread(file[1])
        image_blue = imread(file[2])
        image_nir = imread(file[3])
        # 读取标签图像
        mask = imread(mask)
        # 合并通道
        image = np.stack((image_red, image_green, image_blue, image_nir), axis=-1)
        image = resize(image, (self.img_rows, self.img_cols), preserve_range=True, mode='symmetric')
        mask = resize(mask, (self.img_rows, self.img_cols), preserve_range=True, mode='symmetric')
        # 给标签图像增加通道维度
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)

        # 自适应归一化图像为[0, 1]范围
        image = rescale_intensity(image, out_range=(0, 1))
        mask = rescale_intensity(mask, out_range=(0, 1))

        # 数据增强
        rice = random.randint(1, 15)
        if rice == 7:
            image, mask = flip_horizontal(image, mask)
        elif rice == 8:
            image, mask = flip_vertical(image, mask)
        elif rice == 9:
            image, mask = rotate_image(image, mask)
        elif rice == 10:
            image, mask = adjust_brightness(image, mask)
        elif rice == 11:
            image, mask = add_gaussian_noise(image, mask)
        elif rice == 12:
            image, mask = apply_blur(image, mask)
        elif rice == 13:
            image, mask = apply_occlusion(image, mask)
        else:
            pass

        # numpy数组转换为tensor张量(数据增强过程中会产生负步幅，使用.copy()函数来创建一个正步幅的副本)
        image = torch.tensor(image.copy(), dtype=torch.float32)
        mask = torch.tensor(mask.copy(), dtype=torch.float32)
        # 转换维度顺序:[height, width, channels] 改为 [channels, height, width]
        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)

        return image, mask


class ValDataset(Dataset):
    def __init__(self, zip_list, img_rows, img_cols):
        self.zip_list = zip_list
        self.img_rows = img_rows
        self.img_cols = img_cols

    def __len__(self):
        return len(self.zip_list)

    def __getitem__(self, idx):
        file, mask = self.zip_list[idx]

        image_red = imread(file[0])
        image_green = imread(file[1])
        image_blue = imread(file[2])
        image_nir = imread(file[3])
        mask = imread(mask)

        image = np.stack((image_red, image_green, image_blue, image_nir), axis=-1)
        image = resize(image, (self.img_rows, self.img_cols), preserve_range=True, mode='symmetric')
        mask = resize(mask, (self.img_rows, self.img_cols), preserve_range=True, mode='symmetric')
        # 给标签图像增加通道维度
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)

        # 自适应归一化图像为[0, 1]范围
        image = rescale_intensity(image, out_range=(0, 1))
        mask = rescale_intensity(mask, out_range=(0, 1))
        # numpy数组转换为tensor张量
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        # batch维度在出去后加载时会自动加上，这里不需要加
        # 转换维度顺序:[height, width, channels] 改为 [channels, height, width]
        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)

        return image, mask


class PredictionDataset(Dataset):
    def __init__(self, pre_files, img_rows, img_cols):
        self.pre_files = pre_files
        self.img_rows = img_rows
        self.img_cols = img_cols

    def __len__(self):
        return len(self.pre_files)

    def __getitem__(self, idx):
        file = self.pre_files[idx]
        # print(f"Processing file: {file}")  # 记录当前处理的文件路径

        image_red = imread(file[0])
        image_green = imread(file[1])
        image_blue = imread(file[2])
        image_nir = imread(file[3])

        # # 打印图像尺寸，检查是否形状一致
        # if image_red.shape != image_green.shape:
        #     print(f"R shape: {image_red.shape}, G shape: {image_green.shape},"
        #           f"B shape: {image_blue.shape}, NIR shape: {image_nir.shape}")
        # else:
        #     pass

        image = np.stack((image_red, image_green, image_blue, image_nir), axis=-1)
        image = resize(image, (self.img_rows, self.img_cols), preserve_range=True, mode='symmetric')

        # 自适应归一化图像为[0, 1]范围
        image = rescale_intensity(image, out_range=(0, 1))
        # numpy数组转换为tensor张量
        image = torch.tensor(image, dtype=torch.float32)
        # 转换维度顺序:[height, width, channels] 改为 [channels, height, width]
        image = image.permute(2, 0, 1)

        return image


if __name__ == "__main__":
    test_dataset = PredictionDataset([], 384, 384)
    print("Simple dataset instantiated")

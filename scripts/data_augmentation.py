import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import io, filters, transform
from skimage.exposure import adjust_gamma, rescale_intensity
from skimage.util import random_noise


# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 打印图像像素值
def print_image_stats(image, title):
    print(f"{title} - Min: {image.min()}, Max: {image.max()}, Mean: {image.mean()}")


# 水平翻转(7)
def flip_horizontal(image, mask):
    image_o = np.flip(image, axis=1)
    mask_o = np.flip(mask, axis=1)
    # print(f"Image shape: {image_o.shape}, Mask shape: {mask_o.shape}, 7")
    return image_o, mask_o


# 垂直翻转(8)
def flip_vertical(image, mask):
    image_o = np.flip(image, axis=0)
    mask_o = np.flip(mask, axis=0)
    # print(f"Image shape: {image_o.shape}, Mask shape: {mask_o.shape}, 8")
    return image_o, mask_o


# 旋转(9)
def rotate_image(image, mask):
    angle = random.choice([-10, -20, -30, -40, -50, -60, -70, 0 , 10, 20, 30, 40, 50, 60, 70])
    image_o = transform.rotate(image, angle)
    mask_o = transform.rotate(mask, angle)
    # print(f"Image shape: {image_o.shape}, Mask shape: {mask_o.shape}, 9")
    return image_o, mask_o


# 图像亮度在gamma区间随机均匀拉伸和收缩(10)
def adjust_brightness(image, mask):
    gamma = np.random.uniform(0.5, 1.5)
    image_o = adjust_gamma(image, gamma)
    mask_o = adjust_gamma(mask, gamma)
    # print(f"Image shape: {image_o.shape}, Mask shape: {mask_o.shape}, 10")
    return image_o, mask_o


# 添加标准差为std的高斯噪声(11)
def add_gaussian_noise(image, mask):
    std = np.random.uniform(0.01, 0.07)
    noisy_image_o = random_noise(image, mode='gaussian', var=std)
    noisy_mask_o = random_noise(mask, mode='gaussian', var=std)
    # print(f"Image shape: {noisy_image_o.shape}, Mask shape: {noisy_mask_o.shape}, 11")
    return noisy_image_o, noisy_mask_o


# 添加高斯模糊(12)
def apply_blur(image, mask):
    std = np.random.uniform(0.1, 0.7)
    blurred_image_o = filters.gaussian(image, sigma=std)
    blurred_mask_o = filters.gaussian(mask, sigma=std)
    # print(f"Image shape: {blurred_image_o.shape}, Mask shape: {blurred_mask_o.shape}, 12")
    return blurred_image_o, blurred_mask_o


# 添加遮挡(13)
def apply_occlusion(image, mask):
    occluded_image = image.copy()
    occluded_mask = mask.copy()
    # 检测NumPy数组的维度，并返回h， w的信息
    if image.ndim == 3 and mask.ndim == 3:
        h_image, w_image, _ = image.shape
        h_mask, w_mask, _ = mask.shape
    elif image.ndim == 2 and mask.ndim ==2:
        h_image, w_image = image.shape
        h_mask, w_mask = mask.shape
    else:
        raise ValueError("形状不匹配")

    # 随机位置添加h/8， w/8大小的遮挡
    factor_h = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
    factor_w = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
    occ_image_index = (slice(h_image//8*factor_h, h_image//8*(factor_h+1)),
                       slice(w_image//8*factor_w, w_image//8*(factor_w+1)))
    occ_mask_index = (slice(h_mask//8*factor_h, h_mask//8*(factor_h+1)),
                      slice(w_mask//8*factor_w, w_mask//8*(factor_w+1)))
    # 对三维/二维的数组一定范围像素置为0
    if image.ndim == 3 and mask.ndim == 3:
        occluded_image[occ_image_index[0], occ_image_index[1], :] = 0
        occluded_mask[occ_mask_index[0], occ_mask_index[1], :] = 0
    elif image.ndim == 2 and mask.ndim ==2:
        occluded_image[occ_image_index] = 0
        occluded_mask[occ_mask_index] = 0
    else:
        raise ValueError("形状不匹配")
    # print(f"Image shape: {occluded_image.shape}, Mask shape: {occluded_mask.shape}, 14")
    return occluded_image, occluded_mask


# 处理图像展示
def show_images(images, titles):
    cols = 4  # 每行显示4个图像
    rows = (len(images) + cols - 1) // cols  # 计算行数
    # fig：返回全局属性；axes：返回包含子图对象的数组，可以通过索引访问每个子图
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # 展平子图数组
    for ax, img, title in zip(axes, images, titles):
        # 如果是灰度图，使用灰度颜色映射
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis('off')
    # 隐藏未使用的子图
    for ax in axes[len(images):]:
        ax.axis('off')
    plt.tight_layout()  # 自动调整子图布局
    plt.show()


# 使用示例
if __name__ == "__main__":
    img_path = "../38-Cloud dataset/Predictions/Cloud-Detection_outputs_of_prediction/" \
               "patch_75_4_by_15_LC08_L1TP_002054_20160520_20170324_01_T1.TIF"  # 图像路径
    msk_path = "../38-Cloud dataset/Training/train_gts/" \
               "gt_patch_75_4_by_15_LC08_L1TP_002054_20160520_20170324_01_T1.TIF"

    img = io.imread(img_path)
    msk = io.imread(msk_path)

    # 自适应归一化图像为[0, 1]范围
    img = rescale_intensity(img, out_range=(0, 1))
    msk = rescale_intensity(msk, out_range=(0, 1))

    # 添加翻转和旋转操作
    flipped_h_img, flipped_h_mask = flip_horizontal(img,msk)
    flipped_v_img, flipped_v_mask = flip_vertical(img, msk)
    rotated_img, rotated_mask = rotate_image(img, msk)

    # 添加噪声等
    augmented_img, augmented_mask = adjust_brightness(img, msk)
    noisy_img, noisy_mask = add_gaussian_noise(img, msk)
    blurred_img, blurred_mask = apply_blur(img, msk)
    occluded_img, occluded_mask = apply_occlusion(img, msk)

    # 显示结果
    show_images([img, augmented_img, noisy_img, blurred_img, occluded_img,
                 flipped_h_img, flipped_v_img, rotated_img],
                ['原始图像', '亮度调整图像', '噪声图像', '模糊图像', '遮挡图像',
                 '水平翻转图像', '垂直翻转图像', '旋转图像'])

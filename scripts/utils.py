from tqdm import tqdm


# 定义数据集路径加载类
def get_input_image_names(list_names, directory_name, if_train=True):
    list_img = []
    list_msk = []
    list_test_ids = []

    for filenames in tqdm(list_names['name'], miniters=100):
        nred = 'red_' + filenames
        nblue = 'blue_' + filenames
        ngreen = 'green_' + filenames
        nnir = 'nir_' + filenames

        if if_train:
            dir_type_name = "train"
            fl_img = []
            nmask = 'gt_' + filenames
            fl_msk = directory_name + '/train_gts/' + '{}.TIF'.format(nmask)
            list_msk.append(fl_msk)

        else:
            dir_type_name = "test"
            fl_img = []
            fl_id = '{}.TIF'.format(filenames)
            list_test_ids.append(fl_id)

        fl_img_red = directory_name + '/' + dir_type_name + '_red/' + '{}.TIF'.format(nred)
        fl_img_green = directory_name + '/' + dir_type_name + '_green/' + '{}.TIF'.format(ngreen)
        fl_img_blue = directory_name + '/' + dir_type_name + '_blue/' + '{}.TIF'.format(nblue)
        fl_img_nir = directory_name + '/' + dir_type_name + '_nir/' + '{}.TIF'.format(nnir)
        fl_img.append(fl_img_red)
        fl_img.append(fl_img_green)
        fl_img.append(fl_img_blue)
        fl_img.append(fl_img_nir)

        list_img.append(fl_img)

    if if_train:
        return list_img, list_msk
    else:
        return list_img, list_test_ids


# 定义预测数据集路径加载类
def get_input_image_names_predict(list_names, directory_name):
    list_img = []

    # 读取列表中的文件路径+文件名
    for filenames in tqdm(list_names['name'], miniters=1000):
        # 文件单张路径
        fl_img_single = directory_name + '/' + '{}.TIF'.format(filenames)
        list_img.append(fl_img_single)

    return list_img


# 定义标签数据集路径加载类
def get_input_image_names_mask(list_names, directory_name):
    list_mask = []

    # 读取列表中的文件路径+文件名
    for filenames in tqdm(list_names['name'], miniters=1000):
        n_mask = 'gt_' + filenames

        # 文件单张路径
        fl_mask_single = directory_name + '/train_gts/' + '/' + '{}.TIF'.format(n_mask)
        # 单张存入列表
        list_mask.append(fl_mask_single)

    return list_mask

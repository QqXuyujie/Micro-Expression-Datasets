import os
import shutil

# 设置原始数据集路径和目标路径
dataset_path = r'E:\Xyj_magnify\Xyj_dataset_flownet\Crop_dilb\Crop_SAMM_flow'
output_path = r'E:\Xyj_magnify\Xyj_dataset_flownet\Crop_dilb\subname_samm_crop_flownet_0_5_class'

# 定义五分类
categories = ['Anger', 'Contempt', 'Happiness', 'Other', 'Surprise']

# 遍历原始数据集
for root, dirs, files in os.walk(dataset_path):
    # 获取当前目录中的类别文件夹
    for category in categories:
        if category in dirs:
            category_path = os.path.join(root, category)

            # 在输出路径下创建相应的目录结构
            relative_path = os.path.relpath(category_path, dataset_path)  # 获取相对于原始数据集的路径
            target_category_path = os.path.join(output_path, relative_path)  # 构建目标路径

            # 确保目标目录存在
            if not os.path.exists(target_category_path):
                os.makedirs(target_category_path)

            # 复制文件到新的结构中
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, os.path.join(target_category_path, file))

print("数据集已生成，结构已保留！")

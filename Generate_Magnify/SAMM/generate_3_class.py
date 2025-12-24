import os
import shutil

# 设置原始数据集路径和目标路径
dataset_path = r'E:\Xyj_magnify\Xyj_data_magnify\SAMM\SAMM_Magnify\SAMM_Mgnify_5_amplify'
output_path = r'E:\Xyj_magnify\Xyj_data_magnify\SAMM\SAMM_Magnify\SAMM_Mgnify_3class'

# 定义三分类及其对应的原始类别
category_mapping = {
    'positive': ['Happiness'],
    'negative': ['Sadness', 'Disgust', 'Contempt', 'Fear', 'Anger'],
    'surprise': ['Surprise']
}

# 遍历原始数据集
for root, dirs, files in os.walk(dataset_path):
    # 计算相对路径
    for category, original_categories in category_mapping.items():
        for original_category in original_categories:
            if original_category in dirs:
                original_category_path = os.path.join(root, original_category)

                # 计算子文件夹路径并构建新的目录结构
                relative_dir = os.path.relpath(root, dataset_path)
                target_dir = os.path.join(output_path, relative_dir, category)

                # 创建目标目录（如有需要）
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                # 复制文件
                for file in os.listdir(original_category_path):
                    file_path = os.path.join(original_category_path, file)
                    if os.path.isfile(file_path):
                        shutil.copy(file_path, os.path.join(target_dir, file))

print("数据集已生成，三分类结构已保留！")

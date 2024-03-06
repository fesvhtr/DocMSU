import os
import shutil
import json
from tqdm import tqdm

# 原始文件夹路径
source_folder = '/home/dh/pythonProject/DocMSU/data/img'
destination_folder = '/home/dh/pythonProject/DocMSU/data/release/img'
json_folder = '/home/dh/pythonProject/DocMSU/data/text_data'
json_cnt_dic = {}
# for filename in os.listdir(json_folder):
#     with open(os.path.join(json_folder, filename), 'r', encoding='utf-8') as file:
#         filename = filename.split('.')[0]
#         data_list = []
#         for line in file:
#             # 解析每一行的 JSON 数据
#             json_data = json.loads(line.strip())
#             # 将解析后的数据添加到列表中
#             data_list.append(json_data)
#         json_cnt_dic[filename] = len(data_list)
# print(json_cnt_dic)
# {'technology_nonsar2': 5023, 'education': 274, 'sport_nonsar2': 6599, 'education_nonsar1': 328, 'entertainment_nonsar1': 1234, 'health': 1302, 'environment_nonsar1': 5042, 'sport_nonsar1': 1720, 'science_nonsar1': 1422, 'education_nonsar4': 328, 'education_nonsar3': 5836, 'sport': 835, 'education_nonsar2': 3411, 'technology': 646, 'business': 790, 'health_nonsar2': 5962, 'business_nonsar1': 4446, 'environment': 280, 'entertainment_nonsar3': 6139, 'politic_nonsar2': 3801, 'entertainment_nonsar2': 4997, 'science': 379, 'entertainment': 2022, 'sport_nonsar3': 1720, 'politic': 600, 'politic_nonsar1': 3706, 'technology_nonsar1': 80, 'health_nonsar1': 740, 'health_nonsar3': 740, 'science_nonsar2': 1426}

add_num = {
    'education': 0,
    'education_nonsar1': 274,
    'education_nonsar2': 274 + 328,
    'education_nonsar3': 274 + 328 + 3411,
    'education_nonsar4': 274 + 328 + 3411 + 5836,
    'entertainment': 0,
    'entertainment_nonsar1': 2022,
    'entertainment_nonsar2': 2022 + 1234,
    'entertainment_nonsar3': 2022 + 1234 + 4997,
    'environment': 0,
    'environment_nonsar1': 280,
    'health': 0,
    'health_nonsar1': 1302,
    'health_nonsar2': 1302 + 740,
    'health_nonsar3': 1302 + 740 + 5962,
    'politic': 0,
    'politic_nonsar1': 600,
    'politic_nonsar2': 600 + 3706,
    'science': 0,
    'science_nonsar1': 379,
    'science_nonsar2': 379 + 1422,
    'sport': 0,
    'sport_nonsar1': 835,
    'sport_nonsar2': 835 + 1720,
    'sport_nonsar3': 835 + 1720 + 6599,
    'technology': 0,
    'technology_nonsar1': 646,
    'technology_nonsar2': 646 + 80,
}
for k in tqdm(add_num.keys()):
    for filename in os.listdir(os.path.join(source_folder, k)):
        # 构建原始文件的完整路径
        source_filepath = os.path.join(source_folder,k, filename)
        # 构建新文件的完整路径，将文件名前添加字符 'b'
        new_filename = k.split('_')[0] + '_' + str((int(filename.split('.')[0]) + add_num[k])).zfill(5) + '.jpg'
        destination_filepath = os.path.join(destination_folder, new_filename)

        # 复制文件到新文件夹
        shutil.copy2(source_filepath, destination_filepath)

print("文件已重命名并保存到新文件夹。")
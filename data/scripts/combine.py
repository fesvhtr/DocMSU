import os
import json

destination_folder = '/home/dh/pythonProject/DocMSU/data/release'
text_json_folder = '/home/dh/pythonProject/DocMSU/data/text_data'
img_json_folder = '/home/dh/pythonProject/DocMSU/data/img_label'
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
    'business': 0,
    'business_nonsar1': 790,
}

output_data = {}
"""
{
    'business_1': {
        'sar': 1,
        'text': 'text',
        'label': 'label',
        'img_name': 'business_1.jpg',
        'bbox': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    },
    
"""
for text_file in os.listdir(text_json_folder):
    with open(os.path.join(text_json_folder, text_file), 'r', encoding='utf-8') as file:
        filename = text_file.split('.')[0]
        this_type = filename.split('_')[0]
        if '_nonsar' in filename:
            is_sar = 0
        else:
            is_sar = 1
        for line in file:
            line_data = json.loads(line.strip())
            this_text = line_data['text'].split('\t', 1)[1]
            inner_id = int(line_data['text'].split('.', 1)[0]) + add_num[filename]
            this_img_name = this_type + '_' + str(inner_id).zfill(5) + '.jpg'
            add_dict = {
                'is_sar': is_sar,
                'text': this_text,
                'text_label': line_data['label'],
                'img_name': this_img_name,
                'img_label': None
            }
            output_data[this_type + '_' + str(inner_id).zfill(5)] = add_dict


for img_file in os.listdir(img_json_folder):
    with open(os.path.join(img_json_folder, img_file), 'r', encoding='utf-8') as file:
        filename = img_file.replace('_img', '').split('.')[0]
        this_type = filename.split('_',1)[0]
        for line in file:
            line_data = json.loads(line.strip())
            inner_id = int(line_data['filename'].split('.')[0]) + add_num[filename]
            this_id = this_type + '_' + str(inner_id).zfill(5)
            if this_id in output_data:
                output_data[this_id]['img_label'] = line_data['bbox']
            else:
                print('error' + this_id)

print(output_data['education_00200'])
print(len(output_data))
with open(os.path.join(destination_folder, 'docmsu_all.json'), 'w', encoding='utf-8') as file:
    json.dump(output_data, file, ensure_ascii=False, indent=4)
print('done')
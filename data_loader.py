import jsonlines
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn.functional as F
from torchvision import transforms
import os
import time
from sklearn.model_selection import train_test_split
from PIL import Image
import json


class TextPicPair:
    def __init__(self, filename, imgbox=None, text=None, label=None, textbox=None, img=None, pad=None, cls=None):
        self.filename = filename  # 传入
        self.text = text
        self.imgbox = imgbox  # 传入, [[0,0,0,0,0]]或[[1, x, y, width, height], ...] # 可能一个或多个
        self.label = label  # bbox全为0时没传，bbox正常时传了个1
        self.textbox = textbox
        self.img = img
        self.cls = cls

    def __str__(self):
        return f"TextPicPair(filename={self.filename}, imgbox={self.imgbox}, text={self.text}, label={self.label}, textbox={self.textbox}, img={self.img}, cls={self.cls})"


class STDataset(Dataset):
    def __init__(self, obj_list):
        sentences = []  # 每个元素是一长串字符串
        textboxes = []  # 文本讽刺对象索引[[1, 1]]
        imgboxes = []  # 图片讽刺对象坐标[[0,0,0,0,0]]或[[1, x, y, width, height], ...] # 可能一个或多个
        for obj in obj_list:
            sentences.append(obj.text)
            textboxes.append(obj.textbox)
            imgboxes.append(obj.imgbox[0])  # 这里要注意一个问题，pad后的label[x,y]坐标发生了变化，要根据[width, height]/label判断是否真的是无讽刺的img

        text_mask, textboxes = self.__generate_text_label_mask(textboxes)
        textboxes = [torch.tensor(text_box) for text_box in textboxes]
        imgboxes = [torch.tensor(img_box) for img_box in imgboxes]
        self.ids, self.text_mask, self.textboxes = word2vec(sentences, textboxes, text_mask)
        # self.img_mask, imgboxes = self.__generate_img_label_mask(imgboxes)
        # self.imgs, = torch.stack(imgs, dim=0),   # cat增加现有维度值，stack加新维度
        self.imgboxes = torch.stack(imgboxes, dim=0)
        self.textboxes = torch.stack(textboxes, dim=0)

        self.obj_list = obj_list
        self.transfer_fn = transforms.ToTensor()

    def __generate_text_label_mask(self, textboxes):
        max_length = 0  # 记录最多一段文本中有多少个讽刺对象，education中是4
        for textbox in textboxes:
            length = len(textbox)
            max_length = max(max_length, length)
        b, l = len(textboxes), max_length  # education中b为273表示一共有多少个textbox，其实就是有多少条数据， l为4
        mask = torch.zeros(b, l, dtype=torch.float)
        for row, labels in enumerate(textboxes):  # labels是其中一行文本的textbox，len(labels)表示其中有几段讽刺
            mask[row, :len(labels)] = 1
            label_length = len(labels)
            if label_length < max_length:
                labels += [[-1, -1]] * (max_length - label_length)
            textboxes[
                row] = labels  # textboxes每一行长度为4：例如[[0, 2], [27, 29],[-1, -1], [-1, -1]], [[18], [-1, -1], [-1, -1], [-1, -1]]
        return mask, textboxes  # mask表示每一行有几个真正的讽刺对象，有的话标1， 没的话标0，如：[[1., 0., 0., 0.],[1., 0., 0., 0.],[1., 1., 0., 0.]]

    def __generate_img_label_mask(self, imgboxes):
        max_length = 0  # 记录最多一张图片中最多有多少个讽刺对象，education中是2
        for imgbox in imgboxes:
            length = len(imgbox)
            max_length = max(max_length, length)
        b, l = len(imgboxes), max_length  # education中b为273表示一共有多少个img_box，其实就是有多少条数据， l为2
        mask = torch.ones(b, l, dtype=torch.float)
        for row, labels in enumerate(imgboxes):
            if labels[0][-2:] == [0, 0]:
                imgboxes[row] = [[0, 0, 0, 0, 0]] * l
                mask[row] = 0
            else:
                label_length = len(labels)
                if label_length < max_length:
                    labels += [[0, 0, 0, 0, 0]] * (max_length - label_length)
                    mask[row, label_length:] = 0
                    imgboxes[row] = labels
            imgboxes[row] = torch.tensor(imgboxes[row])
        return mask, imgboxes  # imgboxes里面的每个imgbox长度都相同，且无讽刺的部分均为[0, 0, 0, 0, 0]，mask每一行表示一个imgbox，每一个元素表示imgbox中对应位置的元素是否是讽刺

    def __getitem__(self, idx):
        # 返回：文本对应的idx(带CLS和SEP的), 该句中的attention_mask(填充的为0，原始的为1),...,图像tensor,该图像中的[[label,x,y,width,height], []]
        # 返回一个一维tensor，每个元素代表是否对应的imgbox是讽刺/文本是否有讽刺
        obj = self.obj_list[idx]
        label = obj.cls
        img = cv2.imread(obj.img)
        img_tensor = self.transfer_fn(img)
        img_tensor, img_box = image_pad(img_tensor, obj.imgbox)
        return self.ids[idx], self.text_mask[idx], self.textboxes[idx], img_tensor, self.imgboxes[idx], torch.tensor(
            1), label

    def __len__(self):
        return len(self.obj_list)


def image_pad(image, image_box, max_wh=448):
    # image.shape: [C, H, W], 例如[3, 200, 353], image_box:  [sarcasm , x, y, width, height]
    w, h = image.shape[2], image.shape[1]
    # max_wh = 448   # padding后图片的长、宽
    # left width pad, max_wh - w 看差的是单数还是双数，双数的话直接除以2就是两边的padding数，单数的话左和上多padding一个，
    lwp = int((max_wh - w) / 2) if (max_wh - w) % 2 == 0 else int((max_wh - w) / 2) + 1
    rwp = max_wh - w - lwp
    uhp = int((max_wh - h) / 2) if (max_wh - h) % 2 == 0 else int((max_wh - h) / 2) + 1
    dhp = max_wh - h - uhp
    padding = (lwp, rwp, uhp, dhp)

    label_pad = (0, lwp, uhp, w, h)  # (0, x, y, width, height)
    labels_pad = []
    for box in image_box:
        zipped = zip(box, label_pad)
        label_after = map(sum, zipped)
        labels_pad.append(list(label_after))

    return F.pad(image, padding, value=0, mode='constant'), labels_pad


def word2vec(sentences, labels, mask, if_text_detection=True):
    """
    返回sarcasm数据对应于bert词表中的词向量
    :param sentences: 文本数据集， 一个列表， 每个元素是一长串字符串
    :param mask: 标签掩码, 配合labels使用，每行表明一条文本，该条文本里哪些是有效的讽刺范围(1)哪些是填充(0)
    :param labels: 每条文本对应的讽刺标签，每一行长度都一样，有讽刺标签的范围，也有填充的[-1,-1]
    :return: （Tensor)
        input_ids: 文本词向量
        attention_masks: 每条文本中的实际长度
        labels: 标签词向量
    """
    input_ids = []  # 存储token->idx后的idx列表
    attention_masks = []  # 非pad的特殊字符的token均为1， pad进去的字符为0
    sentences_length = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    for sent in sentences:
        if if_text_detection:
            sent = sent.split()
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=77,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',  # 是否返回tensor类型的value
            truncation=True  # 长度大于max_length时是否截断句子
        )  # 返回的是一个字典，字典中有key: 'input_ids', 'token_type_ids', 'attention_mask'
        sent_len = torch.count_nonzero(encoded_dict['input_ids'], dim=1).reshape(
            -1)  # 分词后句子中不为0的idx的长度(多包含了cls和sep的idx)，这个reshape(-1)好像没什么用？
        sentences_length.append(sent_len)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)  # 3dim -> 2dim, shape:[len(sentences), max_length]
    attention_masks = torch.cat(attention_masks, dim=0)  # 同上
    return input_ids, attention_masks, labels


def read_from_json(json_src,img_dict_path):
    total_obj_list = []
    with open(json_src, 'r+', encoding='utf-8') as f:
        docmsu_data = json.load(f)
    for k, v in docmsu_data.items():
        is_sar = v['is_sar']
        text = v['text']
        text_label = v['text_label']
        img_label = v['img_label']  # 检测的讽刺对象的bounding box坐标，如[{"uuid":"7cd348f6-7438-4e71-845c-efb82035836b","x":42.0174007803,"y":4.0852355957,"width":290.0023567024,"height":194.0,"label":"person"}]
        # bbox为列表，里面是嵌套的字典，包含x,y,width,height, label等key，列表里可能有多个字典(多个讽刺对象)
        filename = v['img_name']
        data_type = v['type']

        # 如果没有标注bbox，说明图片无讽刺
        if len(img_label) == 0:
            fakebox = [[0, 0, 0, 0, 0]]  # [sar, x, y, width, height]   # sar表示sarcasm，0表示无讽刺，1表示有讽刺
            obj = TextPicPair(filename=filename, imgbox=fakebox)  # 创建一个对象
        else:
            imgbox = []
            for val in img_label:
                x, y, width, height, label = val['x'], val['y'], val['width'], val['height'], val['label']
                box = [1, x, y, width, height]  # [sar, x, y, width, height]
                imgbox.append(box)
            obj = TextPicPair(filename=filename, imgbox=imgbox, label=1)
        obj.text = text

        # 处理文本标签
        if len(text_label) != 0:
            obj.textbox = [[label[0], label[1]] if len(label) == 3 else [label[0], label[0]] for label in
                           text_label]  # 这里有一个问题，就是text里的类别没有用到
            obj.cls = torch.tensor(1)  # 有讽刺
        else:
            obj.textbox = [[-1, -1]]  # 图像里的填充标签是0， 文本是-1
            obj.cls = torch.tensor(0)  # 无讽刺
        img_path = os.path.join(img_dict_path, filename)
        obj.img = img_path

        total_obj_list.append(obj)

    return total_obj_list



def load_data(json_src, img_dict_path, batch_size, train_split_ratio=0.7):
    start = time.time()
    res = read_from_json(json_src, img_dict_path)
    temp_label = torch.ones(len(res))
    print(len(res))
    for i in range(10):
        print(res[i])
    res_train, res_test, _, _ = train_test_split(res, temp_label, train_size=train_split_ratio, random_state=0)
    temp_label2 = torch.ones(len(res_test))
    res_val, res_test, _, _ = train_test_split(res_test, temp_label2, train_size=0.66, random_state=0)
    train_iter = DataLoader(STDataset(res_train), batch_size, shuffle=True, drop_last=True)
    val_iter = DataLoader(STDataset(res_val), batch_size)
    test_iter = DataLoader(STDataset(res_test), batch_size)
    end = time.time()
    print(f'加载数据时间：{end - start}')
    print(f'训练集大小：{len(res_train)}, 验证集大小：{len(res_val)}, 测试集大小：{len(res_test)}')
    return train_iter, val_iter, test_iter


if __name__ == '__main__':
    batch_size, epochs = 64, 30

    img_dict_path = "./data/release/img"
    json_src = "./data/release/docmsu_all.json"
    train_iter, val_iter, test_iter = load_data(json_src, img_dict_path, batch_size)

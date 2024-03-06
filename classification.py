from Swin_dh_tokenlevel import swin_tiny_patch4_window7_224
import torch.nn as nn
import torch
from data_loader import load_data
from collections import OrderedDict
from transformers import BertModel
from tqdm import tqdm
from torch.optim import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random
import argparse
# import wandb

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # 设置随机数种子

setup_seed(58)

# wandb.init(project="CESHI",
#            entity="specialone",
#            name="test2_1_5_Seed58_Swin_tiny_classification")

class VisualModel(nn.Module):
    def __init__(self, num_blocks=3):
        super(VisualModel, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(3))
            layers.append(nn.ReLU())
        self.tail_block = nn.Sequential(*layers)
        self.tail_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sTransformer = swin_tiny_patch4_window7_224(num_classes=2)

    def forward(self, x, text_feature):
        x = self.tail_block(x)
        x = self.tail_pool(x)
        output = self.sTransformer(x, text_feature)  # [b, 768]
        return output

class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.Bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768, 96)
        self.fc2 = nn.Linear(77, 49)
        self.act = nn.ReLU()

    def forward(self, ids, mask):
        output = self.Bert(ids, attention_mask=mask)  # output[0] shape:[batch_size, max_len, 768]
        output = self.fc1(output[0]) # [batch_size, 80, 96]
        output = self.act(output)
        output = output.transpose(-2, -1)  # [batch_size, 96, 80]
        output = self.fc2(output)  # [batch_size, 96, 49]
        output = self.act(output)
        output = output.reshape(-1, 96, 7, 7)
        return output

def parse_args():
    parser = argparse.ArgumentParser(description='Your description here')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--img_dict_path', type=str, default='./data/release/img', help='Path to image dictionary')
    parser.add_argument('--json_src', type=str, default='./data/release/docmsu_all.json', help='Path to JSON source file')
    parser.add_argument('--swin_weight', type=str, default='./weights/swin_tiny_patch4_window7_224.pth',
                        help='Path to swin-transformer pretrained weight')
    arser.add_argument('--saved_weight', type=str, default='./weights/',
                       help='Path of dictionary to save docmsu weights')
    return parser.parse_args()

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    train_iter, val_iter, test_iter = load_data(args.json_src, args.img_dict_path, args.batch_size)

    # swin-transformer预训练权重
    weights_dict = torch.load(args.swin_weight, map_location=device)["model"]

    new_weights_dict = OrderedDict()
    for key in weights_dict.keys():
        if "head" in key:
            continue
        new_key = "sTransformer." + key
        new_weights_dict[new_key] = weights_dict[key]

    textmodel = TextModel().to(device)
    visualmodel = VisualModel().to(device)
    visualmodel.load_state_dict(new_weights_dict, strict=False)

    bert_small_params = []
    bert_large_params = []

    for name, parameter in textmodel.named_parameters():
        if name.startswith("fc"):
            bert_large_params += [parameter]
        else:
            bert_small_params += [parameter]


    swin_small_params = []
    swin_large_params = []
    for name, parameter in visualmodel.named_parameters():
        if name.startswith("sTransformer.head") or name.startswith("tail_block"):
            swin_large_params += [parameter]
        else:
            swin_small_params += [parameter]


    params = [
        {"params": bert_small_params, "lr": 5e-6},
        {"params": bert_large_params, "lr": 1e-4},
        {"params": swin_small_params, "lr": 5e-6},
        {"params": swin_large_params, "lr": 1e-4}
    ]

    loss = nn.CrossEntropyLoss()
    optimizer = Adam(params, weight_decay=5e-4)


    for epoch in range(epochs):
        textmodel.train()
        visualmodel.train()
        train_total_loss = 0
        for id, text_mask, text_box, imgs, imgboxes, img_mask, text_label in tqdm(train_iter):  # imgs:[batch_size, 3, 448, 448], imgboxes: [batch_size, 9, 5]
            id, text_mask, text_label = id.to(device), text_mask.to(device), text_label.to(device)
            imgs, imgboxes = imgs.to(device), imgboxes.to(device)
            visual_label = imgboxes[:, 0].long().to(device)  # [batch_size]
            label = torch.max(visual_label, text_label)
            optimizer.zero_grad()
            text_feature = textmodel(id, text_mask)
            output = visualmodel(imgs, text_feature)  # shape:[batch_size, 2]
            l = loss(output, label)
            train_total_loss += l
            l.backward()
            optimizer.step()
        str1 = f"epoch:{epoch},\ttotal_loss:{train_total_loss}"
        print(str1)


        textmodel.eval()
        visualmodel.eval()
        labels_val = []
        predicts_val = []
        total_num_val = 0
        correct_num_val = 0

        labels_test = []
        predicts_test = []
        total_num_test = 0
        correct_num_test = 0

        val_total_loss = 0
        with torch.no_grad():
            for id, text_mask, text_box, imgs, imgboxes, img_mask, text_label in tqdm(val_iter):
                id, text_mask, text_label = id.to(device), text_mask.to(device), text_label.to(device)
                imgs, imgboxes = imgs.to(device), imgboxes.to(device)
                visual_label = imgboxes[:, 0].long().to(device)  # [batch_size]
                label = torch.max(visual_label, text_label)
                total_num_val += len(visual_label)

                text_feature = textmodel(id, text_mask)
                output = visualmodel(imgs, text_feature)
                l = loss(output, label)
                val_total_loss += l
                predict = output.argmax(-1)
                correct_num_val += (predict == label).sum()
                label1 = label.detach().cpu().numpy()
                labels_val.append(label1)
                predict = predict.detach().cpu().numpy()
                predicts_val.append(predict)

            for i, label in enumerate(labels_val):
                if i == 0:
                    continue
                labels_val[0] = np.hstack([labels_val[0], label])

            for j, predict in enumerate(predicts_val):
                if j == 0:
                    continue
                predicts_val[0] = np.hstack([predicts_val[0], predict])

            accuracy = accuracy_score(labels_val[0], predicts_val[0])
            precision = precision_score(labels_val[0], predicts_val[0])
            recall = recall_score(labels_val[0], predicts_val[0])
            f1_score_res = f1_score(labels_val[0], predicts_val[0])

            str2 = f"epoch:{epoch},\tval_total_loss:{val_total_loss}, accuracy:{accuracy:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1 score:{f1_score_res:.4f}\n"
            print(str2)



            for id, text_mask, text_box, imgs, imgboxes, img_mask, text_label in tqdm(test_iter):
                id, text_mask, text_label = id.to(device), text_mask.to(device), text_label.to(device)
                imgs, imgboxes = imgs.to(device), imgboxes.to(device)
                visual_label = imgboxes[:, 0].long().to(device)  # [batch_size]
                label = torch.max(visual_label, text_label)
                total_num_test += len(visual_label)

                text_feature = textmodel(id, text_mask)
                output = visualmodel(imgs, text_feature)

                predict = output.argmax(-1)
                correct_num_test += (predict == label).sum()
                label1 = label.detach().cpu().numpy()
                labels_test.append(label1)
                predict = predict.detach().cpu().numpy()
                predicts_test.append(predict)

            for i, label in enumerate(labels_test):
                if i == 0:
                    continue
                labels_test[0] = np.hstack([labels_test[0], label])

            for j, predict in enumerate(predicts_test):
                if j == 0:
                    continue
                predicts_test[0] = np.hstack([predicts_test[0], predict])

            accuracy = accuracy_score(labels_test[0], predicts_test[0])
            precision = precision_score(labels_test[0], predicts_test[0])
            recall = recall_score(labels_test[0], predicts_test[0])
            f1_score_res = f1_score(labels_test[0], predicts_test[0])

            str3 = f"epoch:{epoch}, accuracy:{accuracy:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1 score:{f1_score_res:.4f}\n"
            print(str3)

        torch.save(visualmodel.state_dict(), f"{args.weights_dir}/visualmodel_{epoch}.pth")
        torch.save(textmodel.state_dict(), f"{args.weights_dir}/textmodel_{epoch}.pth")






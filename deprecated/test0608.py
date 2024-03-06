# import torch
# from chongxie_classification import VisualModel, TextModel
#
# visualmodel = VisualModel()
# visualmodel.load_state_dict(torch.load("./visualmodel_20.pkl"))


from transformers import BertTokenizer, BertModel
bert = BertModel.from_pretrained('bert-base-uncased')
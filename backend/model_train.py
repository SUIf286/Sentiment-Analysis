from model.bert import SentimentBERT
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from transformers import BertTokenizer
from tqdm import trange
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, PYTORCH_PRETRAINED_BERT_CACHE, \
    get_linear_schedule_with_warmup
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score


def read_comments(file_name):
    comments = []
    label = []

    # 读取评论信息
    with open(file_name, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        result = list(reader)

        for line in result[1:]:
            # labels = [int(line[3])+1 ,int(line[4])+1 ,int(line[5])+1 ,int(line[6])+1 ,int(line[7])+1 ,int(line[8])+1 ,int(line[9])+1 ,int(line[10])+1 ,int(line[11])+1 ,int(line[12])+1 ,int(line[13])+1 ,int(line[14])+1 ,int(line[15])+1 ,int(line[16])+1 ,int(line[17])+1 ,int(line[18])+1 ,int(line[19])+1 ,int(line[20])+1 ]
            labels = int(line[3]) + 1
            comments.append(line[1])
            label.append(labels)

    f.close()

    return comments, label

# class SentimentDataset(Dataset):
#         def __init__(self, input_ids, attention_mask, labels,segments):
#             self.input_ids = input_ids
#             self.attention_mask = attention_mask
#             self.labels = labels
#             self.segments = segments
#
#         def __len__(self):
#             return len(self.labels)
#
#         def __getitem__(self, idx):
#             input_id = self.input_ids[idx]
#             # attention_mask = self.attention_mask[idx]
#             label = self.labels[idx]
#             segments = self.segments[idx]
#             # attention_mask
#             return input_id,label,segments


def prepare_data1(comments, tokenizer, max_len):
    input_ids = []
    attention_mask = []
    segments = []
    for comment in comments:
        encoding = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=None,  # 返回 list
        )

        input_ids.append(encoding['input_ids'])  # 移除batch维度
        attention_mask.append(encoding['attention_mask']) # 移除batch维度
        segments.append(encoding['token_type_ids'])

    return input_ids, attention_mask,segments


class Config:
    train_batch_size = 16
    gradient_accumulation_steps = 1
    epochs = 50  # 训练的轮数
    learning_rate = 5e-5
    warmup_proportion = 0.1
    output_dir = 'output'


def train(cfg):
    num_train_steps = int(
        len(train_loader) / cfg.train_batch_size / cfg.gradient_accumulation_steps * cfg.epochs)

    print("***** Running training *****")
    print("Num examples = ", len(train_loader))
    print("Batch size = ", 16)
    print("Num steps = ", num_train_steps)

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentBERT().to(device)

    # 包含不需要衰减的参数，如bias（偏置）和gamma、beta（通常用于BatchNorm层）
    no_decay = ['bias', 'gamma', 'beta']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    # 学习率调度器
    total_steps = len(train_loader) * cfg.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * cfg.warmup_proportion,
                                                num_training_steps=total_steps)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 开始训练
    model.train()
    best_score = 0
    flags = 0

    for _ in trange(int(cfg.epochs), desc="Epoch"):
        # 每一轮的总损失
        total_epoch_loss = 0
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, segments, label,attention_mask = batch

            # 前向传播
            outputs = model(input_ids, segments,attention_mask)
            # loss = outputs.loss
            loss = loss_fn(outputs, label)

            total_epoch_loss = total_epoch_loss + loss.item()
            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if step % 5 == 0:
                print(f"Epoch {_ + 1}/{cfg.epochs}, Iter: {step}/{len(train_loader)}, "
                      f"Train Loss: {loss}, Best F1 Score: {best_score}")

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        print(f"total_loss{total_epoch_loss}")

        print("valid...")
        # val_loss, f1 = val(model,, device)
        model.eval()
        predict = np.zeros((0,), dtype=np.int32)
        gt = np.zeros((0,), dtype=np.int32)
        total_val_loss = 0
        for input_ids, segment_ids, labels ,attention_mask in test_loader:
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = labels.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = model(input_ids, segment_ids,attention_mask)
                # logits = outputs.logits
                val_loss = loss_fn(outputs, label_ids)
                total_val_loss = total_val_loss + val_loss.item()
                pred = outputs.max(1)[1]
                predict = np.hstack((predict, pred.cpu().numpy()))
                gt = np.hstack((gt, label_ids.cpu().numpy()))
                # print(f"predict:{pred}")

        print(f"loss:{val_loss}")

        print("data size:", len(gt))
        # 计算准确率
        accuracy = accuracy_score(gt, predict)
        f1 = np.mean(metrics.f1_score(predict, gt, average=None))

        # return outputs.loss, f1
        print(f"Val Loss: {total_val_loss}, Val F1: {f1},acc:{accuracy}")

        if f1 > best_score:
            best_score = f1
            flags = 0
            # checkpoint = {
            #     'state_dict': model.state_dict()
            # }
            # print(f"saving model to [{cfg.output_dir}] ...")
            # torch.save(checkpoint, args.model_save_pth)
            torch.save(model, "model_11.pth")
            # # tokenizer.save_pretrained(cfg.output_dir)
            # model.save_pretrained(cfg.output_dir)
        else:
            flags += 1
            if flags >= 6:
                break


if __name__ == "__main__":

    # test_data = [
    # "这个手机的电池续航很差，用了不到一天就需要充电了，真是失望。",
    # "天气还算不错，虽然有些阴云，但气温适宜，出行还可以。",
    # "这次旅行非常愉快，景点都很美，导游也很专业，真的是一次难忘的经历。",
    #
    # "服务员态度不太好，等了很久才上菜，而且还冷掉了。",
    # "这本书的内容有点枯燥，虽然有一些有用的知识，但阅读起来不太轻松。",
    # "这家店的装修挺特别的，但价格偏高，性价比一般。",
    #
    # "昨天下了大雨，路面很湿滑，出门的时候差点摔倒。",
    # "今天心情还不错，和朋友一起喝咖啡聊了很久，度过了一个轻松的下午。",
    # "买的衣服非常合身，布料舒服，穿起来特别有气质，很喜欢！"
    # ]
    #
    # test_label = [0, 1, 2, 0, 1, 1, 0, 2, 2]
    #
    # train_data = [
    # "这个产品完全不符合我的期待，质量差，功能也不好，真是浪费钱。",
    # "服务态度差，打电话都没人接，问题一直没有解决。",
    # "今天的天气很糟糕，外面一直在下雨，心情也不好。",
    #
    # "这家餐厅的菜品还可以，服务也算不错，就是价格偏贵了一点。",
    # "今天去健身房锻炼了一下，虽然有点累，但总体感觉还行。",
    # "电影看得还不错，情节不算特别新颖，但也没有让我失望。",
    #
    # "这家店的东西真不错，包装精美，使用起来非常方便，值得推荐！",
    # "今天阳光明媚，心情也特别好，和朋友一起出去散步，度过了愉快的一天。",
    # "这次购物体验非常棒，客服非常耐心，商品质量超出预期，下次还会再来。"
    # ]
    #
    # train_label = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    # 创建模型保存目录
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # # SHUJI
    train_path = './data/train/train_clean.csv'
    test_path = './data/test/test_clean.csv'


    path = r'E:\pytorch\a_whl\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(path)

    train_data, train_label = read_comments(train_path)
    test_data, test_label = read_comments(test_path)

    # 先处理tokenizer
    train_input_ids, train_attention_mask,train_segments= prepare_data1(train_data, tokenizer, max_len=512)
    test_input_ids, test_attention_mask,test_segments= prepare_data1(test_data, tokenizer, max_len=512)

    # TRAIN FEATURE
    train_input_ids = torch.tensor([f  for f in train_input_ids], dtype=torch.long)
    train_attention_mask = torch.tensor([f  for f in train_attention_mask], dtype=torch.bool)
    train_segments = torch.tensor([f  for f in train_segments], dtype=torch.long)
    train_label = torch.tensor([f  for f in train_label], dtype=torch.long)

    # TEST FEATURE
    test_input_ids = torch.tensor([f for f in test_input_ids], dtype=torch.long)
    test_attention_mask = torch.tensor([f for f in test_attention_mask], dtype=torch.bool)
    test_segments = torch.tensor([f for f in test_segments], dtype=torch.long)
    test_label = torch.tensor([f for f in test_label], dtype=torch.long)

    train_data = TensorDataset(train_input_ids, train_segments, train_label,train_attention_mask)
    test_dataset = TensorDataset(test_input_ids, test_segments,test_label,test_attention_mask)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    print('data process finished')

    train(cfg)








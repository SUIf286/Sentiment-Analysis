import numpy as np
from transformers import BertModel,BertTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader ,Dataset
import  pandas as pd
from functools import partial
import time
from torch.optim import AdamW
from torchmetrics import Accuracy
import json


pretrain_path = r'E:\pytorch\a_whl\bert-base-chinese'
target_cols=['Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find','Service#Queue', 'Service#Hospitality','Service#Parking','Service#Timely','Price#Level','Price#Cost_effective','Price#Discount','Ambience#Decoration','Ambience#Noise','Ambience#Space','Ambience#Sanitary','Food#Portion','Food#Taste','Food#Appearance','Food#Recommend'
]

tokenizer1 = BertTokenizer.from_pretrained(pretrain_path)

class RoleDataset(Dataset):
  def __init__(self, mode='train',trans_func=None):
  #定义RoleDataset类的初始化方法，其中mode参数用于指定是训练数据还是测试数据，trans_func参数用于指定数据预处理函数。
      super(RoleDataset, self).__init__()

      if mode == 'train':
          self.data = pd.read_csv('./data/train.csv')
      else:
          self.data = pd.read_csv('./data/test.csv')
      # 获取所有的文本
      self.texts=self.data['review'].tolist()
      # 获取标签
      self.labels=self.data[target_cols].to_dict('records')
      # 将数据集中的标签数据转换为字典格式。
      self.trans_func=trans_func
      # 将数据预处理函数保存到trans_func属性中。

  def __getitem__(self, index):
   #定义数据集类的__getitem__方法，该方法根据给定的索引返回单个样本。
      text=str(self.texts[index])
      label=self.labels[index]
      sample = {
          'text': text
      }
      for label_col in target_cols:
          sample[label_col] =label[label_col]
      # 创建一个字典对象，存入所有的‘text’和‘label’
      sample=self.trans_func(sample)
      # 使用预处理函数trans_func处理样本字典
      return sample

  def __len__(self):
      # 定义数据集类的__len__方法，该方法返回数据集的长度。
      return len(self.texts)

# 转换成id的函数
def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
  # print(example)
  sample={}
  # 使用BERT Tokenizer对输入文本进行编码并生成对应的token ID，类型ID和注意力掩码，并将它们保存在样本字典中。
  encoded_inputs = tokenizer(text=example["text"], add_special_tokens=True,padding='max_length',max_length=max_seq_length,truncation=True,return_attention_mask=True)
  sample['input_ids'] = encoded_inputs["input_ids"]
  sample['token_type_ids'] = encoded_inputs["token_type_ids"]
  sample['attention_mask']   = encoded_inputs["attention_mask"]

  # 将样本的情感标签转换成np.array格式，将所有信息存储到一个新的字典sample中
  sample['Location#Transportation'] = np.array(example["Location#Transportation"], dtype="float32")
  sample['Location#Downtown'] = np.array(example["Location#Downtown"], dtype="float32")
  sample['Location#Easy_to_find'] = np.array(example["Location#Easy_to_find"], dtype="float32")

  sample['Service#Queue'] = np.array(example["Service#Queue"], dtype="float32")
  sample['Service#Hospitality'] = np.array(example["Service#Hospitality"], dtype="float32")
  sample['Service#Parking'] = np.array(example["Service#Parking"], dtype="float32")
  sample['Service#Timely'] = np.array(example["Service#Timely"], dtype="float32")

  sample['Price#Level'] = np.array(example["Price#Level"], dtype="float32")
  sample['Price#Cost_effective'] = np.array(example["Price#Cost_effective"], dtype="float32")
  sample['Price#Discount'] = np.array(example["Price#Discount"], dtype="float32")

  sample['Ambience#Decoration'] = np.array(example["Ambience#Decoration"], dtype="float32")
  sample['Ambience#Noise'] = np.array(example["Ambience#Noise"], dtype="float32")
  sample['Ambience#Space'] = np.array(example["Ambience#Space"], dtype="float32")
  sample['Ambience#Sanitary'] = np.array(example["Ambience#Sanitary"], dtype="float32")

  sample['Food#Portion'] = np.array(example["Food#Portion"], dtype="float32")
  sample['Food#Taste'] = np.array(example["Food#Taste"], dtype="float32")
  sample['Food#Appearance'] = np.array(example["Food#Appearance"], dtype="float32")
  sample['Food#Recommend'] = np.array(example["Food#Recommend"], dtype="float32")

  return sample


trans_func = partial(
      convert_example,
      tokenizer=tokenizer1,
      )

train_ds=RoleDataset('train',trans_func)
test_ds=RoleDataset('test',trans_func)



class EmotionClassifier(nn.Module):
  def __init__(self,n_classes,):
      super(EmotionClassifier, self).__init__()
      self.bert = BertModel.from_pretrained(pretrain_path)
      self.bert_hidden_size = 768

      # Location
      self.out_L_Transportation = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_L_Downtown = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_L_Easy_to_find = nn.Linear(self.bert_hidden_size, n_classes)

      # Service
      self.out_S_Queue = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_S_Hospitality = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_S_Parking = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_S_Timely = nn.Linear(self.bert_hidden_size, n_classes)

        # Price
      self.out_P_Level = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_P_Cost_effective = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_P_Discount = nn.Linear(self.bert_hidden_size, n_classes)
        # Ambience
      self.out_A_Decoration = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_A_Noise = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_A_Space = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_A_Sanitary = nn.Linear(self.bert_hidden_size, n_classes)
        # Food
      self.out_F_Portion = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_F_Taste = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_F_Appearance = nn.Linear(self.bert_hidden_size, n_classes)
      self.out_F_Recommend = nn.Linear(self.bert_hidden_size, n_classes)


  def forward(self, input_ids, token_type_ids,attention_mask):

      outputs = self.bert(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
      pooled_output = outputs.pooler_output

      Transportation = self.out_L_Transportation(pooled_output)
      Downtown       = self.out_L_Downtown(pooled_output)
      Easy_to_find   = self.out_L_Easy_to_find(pooled_output)

      Queue      = self.out_S_Queue(pooled_output)
      Hospitality= self.out_S_Hospitality(pooled_output)
      Parking    = self.out_S_Parking(pooled_output)
      Timely     = self.out_S_Timely(pooled_output)

      Level         = self.out_P_Level(pooled_output)
      Cost_effective= self.out_P_Cost_effective(pooled_output)
      Discount      = self.out_P_Discount(pooled_output)

      Decoration =  self.out_A_Decoration(pooled_output)
      Noise      = self.out_A_Noise(pooled_output)
      Space      = self.out_A_Space(pooled_output)
      Sanitary   = self.out_A_Sanitary(pooled_output)

      Portion    =  self.out_F_Portion(pooled_output)
      Taste      =  self.out_F_Taste(pooled_output)
      Appearance =  self.out_F_Appearance(pooled_output)
      Recommend  =  self.out_F_Recommend(pooled_output)


      return {
          'Transportation': Transportation, 'Downtown': Downtown, 'Easy_to_find': Easy_to_find,
          'Queue': Queue, 'Hospitality': Hospitality, 'Parking': Parking,'Timely':Timely,
          'Level':Level, 'Cost_effective':Cost_effective,'Discount':Discount,
          'Decoration':Decoration,'Noise':Noise,'Space':Space,'Sanitary':Sanitary,
          'Portion':Portion,'Taste':Taste,'Appearance':Appearance,'Recommend':Recommend
      }
      # 接收两个参数input_ids和token_type_ids。在函数内，首先调用bert模型对输入进行处理，得到模型的输出pooled_output。然后分别将pooled_output输入到六个线性层中，得到对六种情感的分类结果。最后将这六个分类结果以字典的形式返回。

# class_names=[1]
model = EmotionClassifier(n_classes=3)# 创建了一个EmotionClassifier的实例model






import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

def collate_func(batch_data):
  # 获取batch数据的大小
  batch_size = len(batch_data)
  # 如果batch_size为0，则返回一个空字典
  if batch_size == 0:
      return {}
  input_ids_list, attention_mask_list,attention_masklist = [], [],[]

  Transportation_list,Downtown_list,Easy_to_find_list=[],[],[]
  Queue_list,Hospitality_list,Parking_list,Timely_list=[],[],[],[]
  Level_list,Cost_effective_list,Discount_list=[],[],[]
  Decoration_list,Noise_list,Space_list,Sanitary_list=[],[],[],[]
  Portion_list,Taste_list,Appearance_list,Recommend_list=[],[],[],[]
  # 遍历batch数据，将每一个数据，转换成tensor的形式
  for instance in batch_data:
      input_ids_temp = instance["input_ids"]
      attention_mask_temp = instance["token_type_ids"]
      attention_mask_temp1 = instance['attention_mask']

    # torch.long 等同于 int64
      input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
      attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
      attention_masklist.append(torch.tensor(attention_mask_temp1,dtype=torch.long))

      Transportation=instance['Location#Transportation']
      Transportation_list.append(torch.tensor(Transportation,dtype=torch.long))

      Downtown=instance['Location#Downtown']
      Downtown_list.append(torch.tensor(Downtown,dtype=torch.long))

      Easy_to_find=instance['Location#Easy_to_find']
      Easy_to_find_list.append(torch.tensor(Easy_to_find,dtype=torch.long))

      Queue=instance['Service#Queue']
      Queue_list.append(torch.tensor(Queue,dtype=torch.long))

      Hospitality=instance['Service#Hospitality']
      Hospitality_list.append(torch.tensor(Hospitality,dtype=torch.long))

      Parking=instance['Service#Parking']
      Parking_list.append(torch.tensor(Parking,dtype=torch.long))

      Timely=instance['Service#Timely']
      Timely_list.append(torch.tensor(Timely,dtype=torch.long))

      Level=instance['Price#Level']
      Level_list.append(torch.tensor(Level,dtype=torch.long))

      Cost_effective=instance['Price#Cost_effective']
      Cost_effective_list.append(torch.tensor(Cost_effective,dtype=torch.long))

      Discount=instance['Price#Discount']
      Discount_list.append(torch.tensor(Discount,dtype=torch.long))

      Decoration=instance['Ambience#Decoration']
      Decoration_list.append(torch.tensor(Decoration,dtype=torch.long))

      Noise=instance['Ambience#Noise']
      Noise_list.append(torch.tensor(Noise,dtype=torch.long))

      Space=instance['Ambience#Space']
      Space_list.append(torch.tensor(Space,dtype=torch.long))

      Sanitary=instance['Ambience#Sanitary']
      Sanitary_list.append(torch.tensor(Sanitary,dtype=torch.long))

      Portion=instance['Food#Portion']
      Portion_list.append(torch.tensor(Portion,dtype=torch.long))

      Taste=instance['Food#Taste']
      Taste_list.append(torch.tensor(Taste,dtype=torch.long))

      Appearance=instance['Food#Appearance']
      Appearance_list.append(torch.tensor(Appearance,dtype=torch.long))

      Recommend=instance['Food#Recommend']
      Recommend_list.append(torch.tensor(Recommend,dtype=torch.long))

    # # 对序列进行padding
    # input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    # attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

  # 对一个batch内的数据，进行padding
  return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
          "token_type_ids": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
          "attention_mask": pad_sequence(attention_masklist, batch_first=True, padding_value=0),

          "Transportation": torch.stack(Transportation_list,dim=0),
          "Downtown": torch.stack(Downtown_list,dim=0),
          "Easy_to_find": torch.stack(Easy_to_find_list,dim=0),

          "Queue": torch.stack(Queue_list,dim=0),
          "Hospitality": torch.stack(Hospitality_list,dim=0),
          "Parking": torch.stack(Parking_list,dim=0),
          "Timely":torch.stack(Timely_list,dim=0),

          "Level": torch.stack(Level_list,dim=0),
          "Cost_effective": torch.stack(Cost_effective_list,dim=0),
          "Discount": torch.stack(Discount_list,dim=0),

          "Decoration": torch.stack(Decoration_list,dim=0),
          "Noise": torch.stack(Noise_list,dim=0),
          "Space": torch.stack(Space_list,dim=0),
          "Sanitary":torch.stack(Sanitary_list,dim=0),

          "Portion": torch.stack(Portion_list,dim=0),
          "Taste": torch.stack(Taste_list,dim=0),
          "Appearance": torch.stack(Appearance_list,dim=0),
          "Recommend":torch.stack(Recommend_list,dim=0),
          }




# 创建一个数据加载器（DataLoader），用于从数据集（dataset）中加载数据并组成一个个批次进行训练或预测
def create_dataloader(dataset, batch_size=1, num_workers=0):

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_func,
        num_workers=num_workers
    )


train_data_loader = create_dataloader(
      train_ds,
      batch_size=4,
      ) #  表示用于batch数据的函数，这里传入了之前定义的collate_func函数。

test_data_loader = create_dataloader(
      test_ds,
      batch_size=4,
      ) #  表示用于batch数据的函数，这里传入了之前定义的collate_func函数。

learning_rate=1e-5
# 设置训练参数
num_train_epochs = 20



decay_params = [
  p.name for n, p in model.named_parameters()
  if not any(nd in n for nd in ["bias", "norm"])
]

# 定义AdamW优化器
optimizer = AdamW(
    [{'params': [p for n, p in model.named_parameters() if n in decay_params], 'weight_decay': 0.01},
     {'params': [p for n, p in model.named_parameters() if n not in decay_params], 'weight_decay': 0.0}],
    lr=learning_rate
)

# 交叉熵损失
criterion = nn.CrossEntropyLoss()




def move_dict_to_device(sample, device):
    """
    将字典中的所有张量移动到指定设备。

    参数:
        sample (dict): 包含张量和其他类型的字典。
        device (torch.device): 目标设备，如 'cuda' 或 'cpu'。

    返回:
        dict: 所有张量已移动到指定设备的新字典。
    """
    moved_sample = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            moved_sample[key] = value.to(device)
        elif isinstance(value, dict):
            # 如果值本身是一个字典，递归处理
            moved_sample[key] = move_dict_to_device(value, device)
        else:
            # 对于非张量的值，保持不变
            moved_sample[key] = value
    return moved_sample


# 假设target_cols包含所有目标列名
cols_list = ['Transportation', 'Downtown', 'Easy_to_find', 'Queue', 'Hospitality', 'Parking', 'Timely',
               'Level', 'Cost_effective', 'Discount', 'Decoration', 'Noise', 'Space', 'Sanitary',
               'Portion', 'Taste', 'Appearance', 'Recommend']

import os




def do_train( model, data_loader,  criterion,  optimizer,device):
    model.train()
    model.to(device)
    # 表示将模型设置为训练模式
    global_step = 0
    # 表示全局训练步数，用于控制训练过程中一些指标的输出频率。
    tic_train = time.time()
    # 表示开始训练的时间点，用于计算训练速度。
    log_steps=10
    best_loss = 8
    save_path = 'train_results.json'
    # 表示训练过程中每隔多少步输出一次训练指标。
    for epoch in range(num_train_epochs):
          losses = []
          # 初始化每个任务的Accuracy计算器
          accuracy_metrics = {col: Accuracy(task='multiclass', num_classes=3).to(device) for col in cols_list}
          for step,sample in enumerate(data_loader):
              #sample.to(device)
              # print(sample)
              sample = move_dict_to_device(sample, device)
              # 表示从样本中获取 input_ids 和 token_type_ids。
              input_ids = sample["input_ids"]
              token_type_ids = sample["token_type_ids"]
              attention_mask = sample["attention_mask"]

              # 表示使用模型进行前向计算，得到预测结果。
              outputs = model(input_ids=input_ids,
                  token_type_ids=token_type_ids,attention_mask=attention_mask)

              loss_sum = 0
              for label_col in cols_list:
                  loss = criterion(outputs[label_col], sample[label_col])
                  loss_sum += loss

                  # 更新相应的Accuracy计算器
                  preds = torch.argmax(outputs[label_col], dim=1)
                  accuracy_metrics[label_col].update(preds, sample[label_col])
              # print(outputs)

              loss_sum.backward()
              optimizer.step()
              optimizer.zero_grad()  # 清空梯度

              losses.append(loss_sum.item())
              global_step += 1

              if global_step % log_steps == 0:
                  accs = {col: metric.compute().item() for col, metric in accuracy_metrics.items()}
                  print("global step %d, epoch: %d, batch: %d, loss: %.8f, accuracies: %s, speed: %.2f step/s" %
                        (len(train_data_loader), epoch+1, step, loss_sum.item(), accs, log_steps / (time.time() - tic_train)))
                  tic_train = time.time()
                  result = {
                      "total_step":len(train_data_loader),
                      "epoch":epoch+1,
                      "step" : step,
                      "loss" : loss_sum.item(),
                      "accuracy": accs
                  }
                  # 保存结果到 JSON 文件
                  with open('train_results.json', 'a') as f:  # 追加模式
                      f.write(json.dumps(result) + '\n')


          # 每个epoch结束后计算并打印准确率
          epoch_accuracies = {col: metric.compute().item() for col, metric in accuracy_metrics.items()}
          avg_loss =  np.mean(losses)

          if avg_loss < best_loss:
              best_loss = avg_loss
              torch.save(model.state_dict(), f"best_model_{epoch+1}.pth")

          print(f"Epoch {epoch + 1} 平均损失{avg_loss:.8f} Accuracies per task: {epoch_accuracies}")
          # 打印当前 epoch 的结果
          print_result = {
              "epoch": epoch + 1,
              "average_loss": avg_loss,
              "accuracies_per_task": epoch_accuracies
          }
          # 保存结果到 JSON 文件
          with open('train_results.json', 'a') as f:  # 追加模式
              f.write(json.dumps(print_result) + '\n')


          # 在每个epoch结束后重置Accuracy计算器
          for metric in accuracy_metrics.values():
              metric.reset()

def check_tensor_device(data, prefix=""):
    if isinstance(data, torch.Tensor):
        print(f"{prefix} tensor is on device: {data.device}")
    elif isinstance(data, dict):
        for key, value in data.items():
            check_tensor_device(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(data, (list, tuple)):
        for idx, item in enumerate(data):
            check_tensor_device(item, f"{prefix}[{idx}]")

# # 在训练或评估循环中每次迭代时调用此函数
# for step, sample in enumerate(data_loader):
#     check_tensor_device(sample)
#     break  # 只检查第一个批次
def evaluate(data_loader,device,cols_list):

    model_eval = EmotionClassifier(n_classes=3).to(device)

    # 加载模型状态字典
    state_dict = torch.load('best_model_20.pth')

    # 将状态字典加载到模型中
    model_eval.load_state_dict(state_dict)

    """Evaluate the model on a validation dataset and calculate accuracy."""
    model_eval.eval()

    # 初始化每个任务的Accuracy计算器
    accuracy_metrics = {col: Accuracy(task='multiclass', num_classes=3).to(device) for col in cols_list}

    with torch.no_grad():  # 关闭梯度计算
        for step, sample in enumerate(data_loader):
            sample = move_dict_to_device(sample, device)
            # check_tensor_device(sample)

            # 表示从样本中获取 input_ids 和 token_type_ids。
            input_ids = sample["input_ids"]
            token_type_ids = sample["token_type_ids"]
            attention_mask = sample["attention_mask"]


            outputs = model_eval(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

            # 更新相应的Accuracy计算器
            for col in cols_list:
                preds = torch.argmax(outputs[col], dim=1)
                accuracy_metrics[col].update(preds, sample[col])
            # 打印每个print_every批次的准确率
            if (step + 1) % 20 == 0 or (step + 1) == len(data_loader):
                batch_accuracies = {col: metric.compute().item() for col, metric in accuracy_metrics.items()}
                print(f"Batch {step + 1}/{len(data_loader)} - Accuracies: {batch_accuracies}")


    # 计算并返回每个任务的准确率
    accuracies = {col: metric.compute().item() for col, metric in accuracy_metrics.items()}
    print(accuracies)
    return accuracies

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
evaluate(test_data_loader,device=device,cols_list=cols_list)


























# do_train(model,train_data_loader,criterion,optimizer,device=device)

# # 多任务学习，分别计算不同标签分类器的损失
# loss_Transportation = criterion(outputs['Transportation'], sample['Transportation'])
# loss_Downtown = criterion(outputs['Downtown'], sample['Downtown'])
# loss_Easy_to_find = criterion(outputs['Easy_to_find'], sample['Easy_to_find'])
#
# loss_Queue = criterion(outputs['Queue'], sample['Queue'])
# loss_Hospitality = criterion(outputs['Hospitality'], sample['Hospitality'])
# loss_Parking = criterion(outputs['Parking'], sample['Parking'])
# loss_Timely = criterion(outputs['Timely'], sample['Timely'])
#
# loss_Level = criterion(outputs['Level'], sample['Level'])
# loss_Cost_effective = criterion(outputs['Cost_effective'], sample['Cost_effective'])
# loss_Discount = criterion(outputs['Discount'], sample['Discount'])
#
# loss_Decoration = criterion(outputs['Decoration'], sample['Decoration'])
# loss_Noise = criterion(outputs['Noise'], sample['Noise'])
# loss_Space = criterion(outputs['Space'], sample['Space'])
# loss_Sanitary = criterion(outputs['Sanitary'], sample['Sanitary'])
#
# loss_Portion = criterion(outputs['Portion'], sample['Portion'])
# loss_Taste = criterion(outputs['Taste'], sample['Taste'])
# loss_Appearance = criterion(outputs['Appearance'], sample['Appearance'])
# loss_Recommend = criterion(outputs['Recommend'], sample['Recommend'])

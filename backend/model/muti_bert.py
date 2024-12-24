# coding=utf-8
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
pretrained_model_path= r'E:\pytorch\a_whl\bert-base-chinese'

class TextCNN(nn.Module):
    def __init__(self, input_dim, kernel_initializer=None):
        super(TextCNN, self).__init__()

        # Define three convolutional layers with different kernel sizes (3, 4, 5)
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(input_dim, 256, kernel_size=4, padding='same')
        self.conv3 = nn.Conv1d(input_dim, 256, kernel_size=5, padding='same')

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: [batch_size, maxlen-2, 768] -> [batch_size, 768, maxlen-2]
        x = x.permute(0, 2, 1)

        cnn1 = F.relu(self.conv1(x))  # [batch_size, 256, maxlen-2]
        cnn1 = F.max_pool1d(cnn1, cnn1.size(2)).squeeze(2)  # [batch_size, 256]

        cnn2 = F.relu(self.conv2(x))  # [batch_size, 256, maxlen-2]
        cnn2 = F.max_pool1d(cnn2, cnn2.size(2)).squeeze(2)  # [batch_size, 256]

        cnn3 = F.relu(self.conv3(x))  # [batch_size, 256, maxlen-2]
        cnn3 = F.max_pool1d(cnn3, cnn3.size(2)).squeeze(2)  # [batch_size, 256]

        # Concatenate the features
        cnn_out = torch.cat([cnn1, cnn2, cnn3], dim=1)  # [batch_size, 256*3]
        cnn_out = self.dropout(cnn_out)  # Apply dropout
        return cnn_out


class BERTTextCNNModel(nn.Module):
    def __init__(self, pretrained_model_path, class_nums):
        super(BERTTextCNNModel, self).__init__()

        # Load BERT model and configuration
        self.bert = BertModel.from_pretrained(pretrained_model_path)

        # TextCNN layer
        self.textcnn = TextCNN(input_dim=768)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 + 768, 256)  # Concatenate CLS token and CNN features
        self.fc2 = nn.Linear(256, class_nums)  # Output layer

    def forward(self, input_ids, attention_mask):
        # Pass input through BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_output.last_hidden_state[:, 0, :]  # Get [CLS] token (shape: [batch_size, hidden_size])
        all_token_embedding = bert_output.last_hidden_state[:, 1:-1, :]  # All tokens except [CLS] and [SEP]

        # Pass the embeddings through TextCNN
        cnn_features = self.textcnn(all_token_embedding)

        # Concatenate CLS token features and CNN features
        features = torch.cat([cls_token, cnn_features], dim=1)  # [batch_size, hidden_size + 256*3]

        # Fully connected layers
        x = F.relu(self.fc1(features))
        output = torch.sigmoid(self.fc2(x))

        return output

if __name__ == '__main__':
    text = '状元楼饭店第一次去，因为地理位置优越，在宁波市和义大道高、大、上，里面装修中式，菜是地道的宁波菜，口味纯正，醉泥螺特棒，吃到了小时候的味道，因为去了晚了，在大堂等了一会儿，期间有茶水喝、服务员还与你聊天，到了就餐时生意太好，服务员都是小跑状，服务态度绝对不提速，样样都服务到位，点酒水还耐心的与我们解释，就这样绝对要夸一夸，特别是彭新星、洪继华（看服务牌才知道名字）也给我们宁波市形象增色，状元楼是宁波的一扇窗口，服务员的素质更体现我们宁波人的精神面貌。赞一个'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512,
                         return_tensors='pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoding = {k: v.to(device) for k, v in encoding.items()}

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    model = BERTTextCNNModel(pretrained_model_path, class_nums=18)
    model.to(device)
    output = model(input_ids, attention_mask)
    print(output)






# if __name__ == '__main__':
#     test_data = [
#     "这个手机的电池续航很差，用了不到一天就需要充电了，真是失望。",
#     "天气还算不错，虽然有些阴云，但气温适宜，出行还可以。",
#     "这次旅行非常愉快，景点都很美，导游也很专业，真的是一次难忘的经历。",

#     "服务员态度不太好，等了很久才上菜，而且还冷掉了。",
#     "这本书的内容有点枯燥，虽然有一些有用的知识，但阅读起来不太轻松。",
#     "这家店的装修挺特别的，但价格偏高，性价比一般。",

#     "昨天下了大雨，路面很湿滑，出门的时候差点摔倒。",
#     "今天心情还不错，和朋友一起喝咖啡聊了很久，度过了一个轻松的下午。",
#     "买的衣服非常合身，布料舒服，穿起来特别有气质，很喜欢！"
#     ]

#     test_label = [0, 1, 2, 0, 1, 1, 0, 2, 2]

#     train_data = [
#     "这个产品完全不符合我的期待，质量差，功能也不好，真是浪费钱。",
#     "服务态度差，打电话都没人接，问题一直没有解决。",
#     "今天的天气很糟糕，外面一直在下雨，心情也不好。",

#     "这家餐厅的菜品还可以，服务也算不错，就是价格偏贵了一点。",
#     "今天去健身房锻炼了一下，虽然有点累，但总体感觉还行。",
#     "电影看得还不错，情节不算特别新颖，但也没有让我失望。",

#     "这家店的东西真不错，包装精美，使用起来非常方便，值得推荐！",
#     "今天阳光明媚，心情也特别好，和朋友一起出去散步，度过了愉快的一天。",
#     "这次购物体验非常棒，客服非常耐心，商品质量超出预期，下次还会再来。"
#     ]

#     train_label = [0, 0, 0, 1, 1, 1, 2, 2, 2]


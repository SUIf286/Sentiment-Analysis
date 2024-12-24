import random
import re
from math import sqrt as msqrt
import torch
import torch.functional as F
from torch import nn
from torch.optim import Adadelta
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
# 参数说明
max_len = 512  # 输入序列的最大长度
max_vocab = 21128  # 字典的最大大小
max_pred = 5  # Mask时最大的Mask数量

d_k = d_v = 64  # 自注意力中K和V的维度, Q的维度直接用K的维度代替, 因为这二者必须始终相等
d_model = 768  # n_heads * d_k  # Embedding的大小
d_ff = d_model * 4  # 前馈神经网络的隐藏层大小, 一般是d_model的四倍

n_heads = 8  # 多头注意力的头数.
n_layers = 4  # Encoder的堆叠层数
n_segs = 2  # 输入BERT的句子段数，用于制作Segment Embedding

p_dropout = .1  # BERT中所有dropout的概率
# BERT propability defined
p_mask = .8
p_replace = .1
p_do_nothing = 1 - p_mask - p_replace


# 80%的概率将被选中的单词替换为[MASK].

# 10%的概率将被选中的单词替换为随机词.

# 10%的概率对被选中的单词不进行替换


# 生成填充掩码
def get_pad_mask(tokens, pad_idx=0):
    '''
    suppose index of [PAD] is zero in word2idx
    tokens: [batch, seq_len]
    '''
    batch, seq_len = tokens.size()
    pad_mask = tokens.data.eq(pad_idx).unsqueeze(1)
    pad_mask = pad_mask.expand(batch, seq_len, seq_len)
    return pad_mask


def expand_attention_mask(attention_mask):
    '''
    Expand attention mask from [batch, seq_len] to [batch, seq_len, seq_len]

    :param attention_mask: Tensor of shape [batch, seq_len]
    :return: Expanded attention mask of shape [batch, seq_len, seq_len]
    '''
    batch, seq_len = attention_mask.size()
    # 增加一个维度并在该维度上进行广播
    expanded_mask = attention_mask.unsqueeze(1).expand(batch, seq_len, seq_len)
    return expanded_mask


def gelu(x):
    '''
    Two way to implements GELU:
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    or
    0.5 * x * (1. + torch.erf(torch.sqrt(x, 2)))
    '''
    return .5 * x * (1. + torch.erf(x / msqrt(2.)))


# 编码层
class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()
        self.seg_emb = nn.Embedding(n_segs, d_model)
        self.word_emb = nn.Embedding(max_vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, seg):
        '''
        x: [batch, seq_len]
        '''
        word_enc = self.word_emb(x)

        # positional embedding
        pos = torch.arange(x.shape[1], dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).expand_as(x)
        pos_enc = self.pos_emb(pos)

        seg_enc = self.seg_emb(seg)
        x = self.norm(word_enc + pos_enc + seg_enc)
        return self.dropout(x)
        # return: [batch, seq_len, d_model]


# 点积缩放注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2) / msqrt(d_k))
        # scores: [batch, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # context: [batch, n_heads, seq_len, d_v]
        context = torch.matmul(attn, V)
        return context


# 多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q, K, V, attn_mask):
        '''
        Q, K, V: [batch, seq_len, d_model]
        attn_mask: [batch, seq_len, seq_len]
        '''
        batch = Q.size(0)
        '''
        split Q, K, V to per head formula: [batch, seq_len, n_heads, d_k]
        Convenient for matrix multiply opearation later
        q, k, v: [batch, n_heads, seq_len, d_k / d_v]
        '''
        per_Q = self.W_Q(Q).view(batch, -1, n_heads, d_k).transpose(1, 2)
        per_K = self.W_K(K).view(batch, -1, n_heads, d_k).transpose(1, 2)
        per_V = self.W_V(V).view(batch, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # context: [batch, n_heads, seq_len, d_v]
        context = ScaledDotProductAttention()(per_Q, per_K, per_V, attn_mask)
        context = context.transpose(1, 2).contiguous().view(
            batch, -1, n_heads * d_v)

        # output: [batch, seq_len, d_model]
        output = self.fc(context)
        return output


# ffn 前馈神经网络
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_dropout)
        self.gelu = gelu

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


# transfromer 编码层
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.enc_attn = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()

    def forward(self, x, pad_mask):
        '''
        pre-norm
        see more detail in https://openreview.net/pdf?id=B1x8anVFPr

        x: [batch, seq_len, d_model]
        '''
        residual = x
        x = self.norm1(x)
        x = self.enc_attn(x, x, x, pad_mask) + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x + residual


class SentimentBERT(nn.Module):
    def __init__(self, n_layers=4, num_classes=3):  # 添加num_classes参数，默认值为3
        super(SentimentBERT, self).__init__()
        self.embedding = Embeddings()
        self.encoders = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        # 替换原来的分类器为新的具有3个输出单元的分类器
        self.classifier = nn.Linear(d_model, num_classes)

    # attention_mask
    def forward(self, tokens, segments,attention_mask):
        output = self.embedding(tokens, segments)
        # enc_self_pad_mask  = get_pad_mask(tokens)
        enc_self_pad_mask = expand_attention_mask(attention_mask)
        # print(f"enc_self_pad_mask:{enc_self_pad_mask}")
        for layer in self.encoders:
            output = layer(output, enc_self_pad_mask)

        # cls_output = output[:, 0]

        cls_output = output[:, 0, :]  # [batch, d_model]
        logits = self.classifier(cls_output)  # [batch, num_classes]

        return logits






if __name__ == '__main__':

    from torch.utils.data import Dataset
    import torch
    from transformers import BertTokenizer


    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts  # 文本数据列表
            self.labels = labels  # 标签列表
            self.tokenizer = tokenizer  # BERT tokenizer
            self.max_len = max_len  # 最大序列长度

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):
            text = self.texts[item]
            label = self.labels[item]

            # 使用BERT tokenizer进行分词和编码
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,  # 添加[CLS]和[SEP]
                max_length=self.max_len,  # 最大长度
                padding='max_length',  # 填充到max_length
                truncation=True,  # 截断长于max_len的文本
                return_attention_mask=True,  # 返回注意力掩码
                return_tensors='pt'  # 返回PyTorch的Tensor格式
            )

            input_ids = encoding['input_ids'].flatten()  # 获取input_ids
            attention_mask = encoding['attention_mask'].flatten()  # 获取attention_mask
            token_type_ids = encoding['token_type_ids'].flatten()  # 获取segment_ids

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'labels': torch.tensor(label, dtype=torch.long)  # 标签
            }


    # 示例：加载数据并创建DataLoader
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    model = SentimentBERT()
    # print(model)
    # 假设你有以下文本数据和标签
    texts = ["这家餐厅非常好，服务态度也不错", "食物太难吃了，完全不推荐"]
    labels = [1, -1]  # 假设1表示正面情感，-1表示负面情感
    path = r'E:\pytorch\a_whl\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(path)

    # 创建Dataset
    dataset = SentimentDataset(texts, labels, tokenizer, max_len=512)

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 查看数据
    for batch in dataloader:
        # print(batch)
        # 输入到模型中的数据
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        print(f"attention_mask:{attention_mask}")

        labels = batch['labels']


        logits = model(input_ids, token_type_ids)
        print(logits)

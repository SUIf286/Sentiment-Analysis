from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer,BertModel
import torch.nn as nn


app = FastAPI()

# 加载 tokenizer 和训练好的模型
pretrain_path = r'E:\pytorch\a_whl\bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrain_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(texts, tokenizer, max_len=512):
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt', 
    )
    return encodings['input_ids'], encodings['token_type_ids'], encodings['attention_mask']

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
def predict(model, tokenizer, texts, max_len=512):
    input_ids, token_type_ids,attention_mask = preprocess(texts, tokenizer, max_len)
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask =  attention_mask.to(device)

    # 将输入数据传入模型
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids,attention_mask)


    # 假设模型输出是字典，其中键是任务名（如情感类别）
    # 并且每个值是一个形状为 (batch_size, num_classes) 的张量
    predictions = {col: torch.argmax(output, dim=1).item() for col, output in outputs.items()}
    # # 打印预测结果
    # print("预测结果：")
    # for col, pred_idx in predictions.items():
    #     print(f"{col}: {sentiment_labels[pred_idx]} ({pred_idx})")

    return predictions


model = EmotionClassifier(n_classes=3)
state_dict = torch.load(r'E:\Demo\Sentiment_Analysis\backend\best_model_20.pth', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

class TextRequest(BaseModel):
    text: str

@app.post("/predict", response_model=dict)
async def predict_emotions(request: TextRequest):
    try:
        predictions = predict(model, tokenizer, request.text)
        # 将索引转换为情感标签
        sentiment_labels = ["消极", "正常", "积极"]
        labeled_predictions = {key: sentiment_labels[val] for key, val in predictions.items()}
        return labeled_predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn server:app --reload

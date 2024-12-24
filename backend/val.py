import torch
import torchmetrics
import torch.nn as nn


def predict(model, test_data_loader):
    val_loss = 0
    test_pred = defaultdict(list)
    model.eval()
    for step, batch in tqdm(enumerate(test_data_loader)):
        b_input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']

        with paddle.no_grad():
            logits = model(input_ids=b_input_ids, token_type_ids=token_type_ids)
            for col in target_cols:
                print(col, logits[col])
                out2 = paddle.argmax(logits[col], axis=1)
                # 求出当前col输出下对应的情感级数
                print(out2)
                test_pred[col].extend(out2.numpy().tolist())
                # 将每个col下所有的情感级数按照批次存放到test_pred里面
                print(test_pred[col])

    return test_pred


submit = pd.read_csv('data/submit_example.tsv', sep='\t')
test_pred = predict(model, test_data_loader)

import json
import os
from collections import defaultdict

import numpy as np
import pyndeval
import torch
from models import PairwiseClassificationModel
from dataloader import (OnceTrainDataset, OnceEvalDataset, DataLoader, collote_fn_trained, collote_fn_eval,
                        reformulate_sentence_v1, reformulate_sentence_v2, reformulate_sentence_v3)
# from pairwise_semantic_compare.finetune_model
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import get_scheduler
import torch.nn as nn
from sklearn.metrics import precision_score, f1_score, recall_score
from pyndeval import SubtopicQrel, ScoredDoc


lr = 5e-6
batch_size = 4
train_epochs = 10
device = 'cuda'
data_dir = '../diversification_data/cross_validation/subtopics_suggestions'
fold = 1
select_reformulate_func = reformulate_sentence_v1

model_save_dir = f'./cpkts/{select_reformulate_func.__name__}_fold_{fold}'
os.makedirs(model_save_dir, exist_ok=True)

model = PairwiseClassificationModel().to(device)
optimizer = AdamW(model.parameters(), lr=lr)

train_dataset = OnceTrainDataset(data_dir, fold, reformulate_func=select_reformulate_func)
valid_dataset = OnceEvalDataset(data_dir, fold, reformulate_func=select_reformulate_func)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collote_fn_trained)
valid_dataloader =DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=collote_fn_eval)

loss_fn = nn.BCEWithLogitsLoss()

num_training_steps = train_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.05 * num_training_steps),
    num_training_steps=num_training_steps,
)


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch - 1) * len(dataloader)

    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        y = y.view(-1, 1).float()
        pred = model(**X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']

    run_list = []
    model.eval()
    with torch.no_grad():
        for X, y, qids in tqdm(dataloader):
            X = X.to(device)
            labels = y[0] # batch size == 1
            qid = qids[0]
            pred = model(**X)
            for label, score in zip(labels, torch.sigmoid(pred).detach().cpu().numpy()):
                run_list.append(ScoredDoc(str(qid), label, score))
    dict_result = pyndeval.ndeval(valid_dataset.qrels_all(), run_list, measures=["alpha-nDCG@20"])
    metrics = defaultdict(list)
    for item in dict_result.values():
        for key, value in item.items():
            metrics[key].append(value)
    metrics = {key: float(np.mean(value)) for key, value in metrics.items()}
    print(f"{mode} Accuracy: {metrics}%\n")
    return metrics


total_loss = 0.
total_res = {}
for t in range(train_epochs):
    print(f"Epoch {t+1}/{train_epochs}\n-------------------------------")
    torch.cuda.empty_cache()
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    torch.cuda.empty_cache()
    res = test_loop(valid_dataloader, model, mode='Valid')
    total_res[t] = res
    torch.save(model.state_dict(), os.path.join(model_save_dir, f'epoch_{train_epochs}_nDCG{res["alpha-nDCG@20"]}.pth'))
    with open(os.path.join(model_save_dir, 'eval.score'), 'w') as f:
        json.dump(total_res, f, ensure_ascii=False, indent=4)

print("Done!")


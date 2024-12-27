import os
import json

import torch
from pyndeval import SubtopicQrel
from torch.utils.data import Dataset, DataLoader
from bce import load_bce_embedding, load_bce_rerank
import pickle as pk
from functools import cache
from config import base_model_type


if base_model_type == 'bce_embedding':
    tokenizer, _ = load_bce_embedding()
elif base_model_type == 'bce_rerank':
    tokenizer, _ = load_bce_rerank()


abs_file_path = os.path.split(__file__)[0]


def reformulate_sentence_v1(query, sentence, prompt):
    return f'{query}{tokenizer.sep_token}{sentence}'


def reformulate_sentence_v2(query, sentence, prompt):
    return f'{query}{tokenizer.sep_token}{sentence}{tokenizer.sep_token}{prompt}'


def reformulate_sentence_v3(query, sentence, prompt: str):
    prompt = prompt.replace('Consider a query A and a passage B. The subtopics of this query are as follows:', '')
    prompt = prompt.replace("These subtopics are specific and subdivided topics related to the query. "
                            "Please determine whether passage B contains possible intents related to query A, "
                            "and give the answer by predicting 'yes' or 'no'", '')

    return f'{query}{tokenizer.sep_token}{sentence}{tokenizer.sep_token}{prompt}'


def load_origin_test(data_dir, idx):
    with open(os.path.join(data_dir, f'test_qids_fold{idx}.pkl'), 'rb') as f:
        return pk.load(f)


def load_origin_train(data_dir, idx):
    data = []
    with open(os.path.join(data_dir, f'train_jsonl_fold{idx}.jsonl'), 'r', encoding='utf8') as f:
        for line in f:
            if len(line) > 0:
                data.append(json.loads(line))
    return data


@cache
def get_subtopic_all():
    # 把 qid ，query，subtopic都读进
    filename = "subtopics_suggestions" # if not "intent" in load else "subtopics_intents"
    with open(os.path.join(abs_file_path, f'../diversification_data/{filename}.json'), 'r') as f:
        qid_query_subtopic = json.load(f)
    return qid_query_subtopic


def get_prompt(qid):
    qid_query_subtopic = get_subtopic_all()
    prompt_former = "Consider a query A and a passage B. The subtopics of this query are as follows:"
    prompt_latter = ("These subtopics are specific and subdivided topics related to the query. "
                     "Please determine whether passage B contains possible intents related to query A, "
                     "and give the answer by predicting 'yes' or 'no'")

    subtopics = qid_query_subtopic[qid]['subtopic'].split(",")

    prompt = "" + prompt_former
    for index, subtopic in enumerate(subtopics):
        subtopic = " ".join(subtopic.split("_"))
        number = index + 1
        if number == len(subtopics):
            prompt = prompt + f"{number}.'" + subtopic + "'. "
        else:
            prompt = prompt + f"{number}.'" + subtopic + "', "

    return prompt + prompt_latter


class OnceTrainDataset(Dataset):

    def __init__(self, dir_path, fold, reformulate_func=reformulate_sentence_v1):
        self.dir_path = dir_path
        self.reformulate_func = reformulate_func

        self.data = load_origin_train(dir_path, fold)

    def __len__(self):
        return len(self.data) * 2

    def __getitem__(self, item):
        idx = item // 2
        if item % 2 == 0:
            return {
                'sentence': self.reformulate_func(self.data[idx]['query'], self.data[idx]['pos'], self.data[idx]['prompt']),
                'label': 1,
            }
        else:
            return {
                'sentence': self.reformulate_func(self.data[idx]['query'], self.data[idx]['neg'], self.data[idx]['prompt']),
                'label': 0,
            }

def collote_fn_trained(batch_samples):
    batch_sentence = []
    batch_label = []
    for sample in batch_samples:
        batch_sentence.append(sample['sentence'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y

# with open('../diversification_data/subtopics_suggestions.json','r') as f:
#     query_and_subtopic=json.load(f)


def default_promot_func(query):
    inputs = ("Consider a query A and a passage B. The subtopics of this query are as follows:"
              "{}"
              # "1.'french lick resort and casino job openings', 2.'french lick resort and casino jobs', 3.'french lick resort and casino restaurants', 4.'french lick resort casino address', 5.'french lick resort casino concerts', 6.'french lick resort casino coupon codes', 7.'french lick resort casino coupons', 8.'french lick resort casino entertainment', 9.'french lick resort casino hours'. "
              "These subtopics are specific and subdivided topics related to the query. Please determine whether passage B contains possible intents related to query A, and give the answer by predicting 'yes' or 'no'")


class OnceEvalDataset(Dataset):

    def __init__(self, dir_path, fold, reformulate_func=reformulate_sentence_v1):
        self.dir_path = dir_path
        self.reformulate_func = reformulate_func

        self.label_idx = load_origin_test(dir_path, fold)

        with open(os.path.join(abs_file_path, f'../diversification_data/indri_data_installed.pkl'), 'rb') as f:
            self.data = pk.load(f)

    @cache
    def qrels_all(self):
        qrels_data_path = os.path.join(abs_file_path, f'../diversification_data/qrels_installed.pkl')
        with open(qrels_data_path, 'rb') as f:
            qrels = pk.load(f)
        return [SubtopicQrel(item[0],item[1],item[2],item[3]) for item in qrels]

    def __len__(self):
        return len(self.label_idx)

    def __getitem__(self, item):
        _query_item = item
        query, items = self.data[self.label_idx[item]]
        prompt = get_prompt(self.label_idx[item])
        res = []
        for item in items:
            res.append({
                'sentence': self.reformulate_func(query, item[1], prompt),
                'label': item[0],
            })
        return res, self.label_idx[_query_item]


def collote_fn_eval(batch_samples):
    batch_sentence = []
    batch_label = []
    qids = []
    for sample, qid in batch_samples:
        qids.append(qid)
        for item in sample:
            batch_sentence.append(item['sentence'])
        batch_label.append([item['label'] for item in sample])
    X = tokenizer(
        batch_sentence,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return X, batch_label, qids


if __name__ == '__main__':
    dataset = OnceTrainDataset('../diversification_data/cross_validation/subtopics_suggestions', '1', reformulate_func=reformulate_sentence_v2)
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collote_fn_trained)
    batch_X, batch_y = next(iter(train_dataloader))

    dataset2 = OnceEvalDataset('../diversification_data/cross_validation/subtopics_suggestions', '1', reformulate_func=reformulate_sentence_v2)
    eval_dataloader = DataLoader(dataset2, batch_size=1, shuffle=True, collate_fn=collote_fn_eval)
    batch_X, batch_y, qids = next(iter(eval_dataloader))

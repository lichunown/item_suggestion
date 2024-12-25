from datasets import load_dataset,Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from pyndeval import SubtopicQrel, ScoredDoc
from FlagEmbedding import FlagReranker
from sklearn.metrics import mean_squared_error
import pickle,json,gzip
# 加载数据集
import sys,os
pretrain,save,teacher=sys.argv[1],sys.argv[2],sys.argv[3]
base_path='/root/autodl-tmp/workspace'
indri_data=f'{base_path}/diversification_data/indri_data_installed.pkl'
qrels_data=f'{base_path}/diversification_data/qrels_installed.pkl'
cv_path=f"{base_path}/diversification_data/cross_validation/subtopics_suggestions/"
MAX_LENGTH=128
def read_pickle(path):
    if path.endswith("pkl.gz"):
        with gzip.open(path,'rb') as f:
            data=pickle.load(f)
    else:
        with open(path,'rb') as f:
            data=pickle.load(f)
    return data
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}
def train_model(dataset_train:Dataset,dataset_valid:Dataset,save_path:str):
# 定义训练参数
    model_name = pretrain
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    def preprocess_function(examples):
        return tokenizer(examples["query"],examples["doc"], truncation=True, padding=True)
    training_args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
    )
    dataset_train = dataset_train.map(preprocess_function, batched=True)
    dataset_valid = dataset_valid.map(preprocess_function, batched=True)
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # 开始训练
    trainer.train()
    # 在测试集上进行评估
    predictions = trainer.predict(dataset_valid)
    mse = mean_squared_error(dataset_valid["label"], predictions.predictions.flatten())
    print(f"Test MSE: {mse:.4f}")
    return model
def init_ranker(pretrain_path,save_path):
    ranker=FlagReranker(pretrain_path)
    return ranker
if __name__=="__main__":
    indri_rank_dict=read_pickle(indri_data)
    full_docid_dict={}
    for qid in indri_rank_dict:
        _,doc_list=indri_rank_dict[qid]
        for item in doc_list:
            docid,doc_content=item
            doc_content=doc_content[:MAX_LENGTH]
            full_docid_dict[docid]=doc_content
    qrels=read_pickle(qrels_data)
    qrels=[SubtopicQrel(item[0],item[1],item[2],item[3]) for item in qrels]
    fold_id=[str(i+1) for i in range(5)]
    eval_metrics_overall=[]
    with open(f'{base_path}/diversification_data/subtopics_suggestions.json','r') as f:
        qid_query_subtopic=json.load(f)
    teacher_data=read_pickle(teacher)
    for fold in fold_id:
        test_qids_path=cv_path+f"test_qids_fold{fold}.pkl"
        test_qids=read_pickle(test_qids_path)
        dataset_train=[]
        dataset_test=[]
        for item in teacher_data:
            qid,docid,score=item
            query,_=indri_rank_dict[qid]
            subtopics=qid_query_subtopic[qid]['subtopic'].replace("_"," ")
            full_query=query+" "+subtopics
            doc=full_docid_dict[docid]
            item={"query":full_query,"doc":doc,"label":score}
            if qid in test_qids:
                dataset_test.append(item)
            else:
                dataset_train.append(item)
        dataset_train=Dataset.from_list(dataset_train)
        dataset_test=Dataset.from_list(dataset_test)
        path_save=f"{save}/fold{fold}"
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        model=train_model(dataset_train,dataset_test,path_save)
        break
   
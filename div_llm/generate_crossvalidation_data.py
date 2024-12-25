import numpy as np
import pickle
import json
base_path='/root/autodl-tmp/workspace'
indri_data=f'{base_path}/diversification_data/indri_data_installed.pkl'
qrels_data=f'{base_path}/diversification_data/qrels_installed.pkl'
base_data_path=f"{base_path}/finetune_data_bge_jsonl/subtopic_top100.jsonl"
cv_path=f"{base_path}/diversification_data/cross_validation/subtopics_suggestions_top100/"
from sklearn.model_selection import KFold
def load_pickle(path):
    with open(path,'rb') as f:
        data=pickle.load(f)
    return data
def save_pickle(item,path):
    with open(path,'wb') as f:
        pickle.dump(item,f)
def save_text(lines,path):
    with open(path,'w') as f:
        for line in lines:
            f.write(line+'\n')
def load_text(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]
if __name__=='__main__':
    X=np.arange(200)
    kf=KFold(n_splits=5,shuffle=True,random_state=0)
    cnt=1 
    full_jsons=[json.loads(item) for item in load_text(base_data_path)]
    indri_rank_dict=load_pickle(indri_data)
    for train_index, test_index in kf.split(X):
        train_qids=[str(item+1) for item in train_index.tolist() if str(item+1)!='95' and str(item+1)!='100']
        test_qids=[str(item+1) for item in test_index.tolist() if str(item+1)!='95' and str(item+1)!='100']
        print("train:",train_qids)
        print("test:",test_qids)
        test_qid_path=cv_path+f"test_qids_fold{cnt}.pkl"
        save_pickle(test_qids,test_qid_path)
        train_queries=set([indri_rank_dict[qid][0] for qid in train_qids])
        train_json_list=[json.dumps(item) for item in full_jsons if item["query"] in train_queries]
        train_jsonl_path=cv_path+f"train_jsonl_fold{cnt}.jsonl"
        save_text(train_json_list,train_jsonl_path)
        print(f"fold {cnt} saved.")
        cnt+=1

from FlagEmbedding import FlagLLMReranker
from FlagEmbedding import FlagReranker
from peft import PeftModel
import pickle
import pyndeval
from pyndeval import SubtopicQrel, ScoredDoc
from tqdm import tqdm
import numpy as np
import sys,os
pretrain,save=sys.argv[1],sys.argv[2]
base_path='/root/autodl-tmp/workspace'
indri_data=f'{base_path}/diversification_data/indri_data_installed.pkl'
qrels_data=f'{base_path}/diversification_data/qrels_installed.pkl'
cv_path=f"{base_path}/diversification_data/cross_validation/query_only/"
#pretrain_path=f'{base_path}/gemma-2b'
#pretrain_path=f'{base_path}/bge-reranker-v2-m3'

pretrain_path=f"/root/autodl-tmp/pretrained_models/{pretrain}"
peft_path=f'{base_path}/save_path/{save}/fold'
MAX_LENGTH=512
#MAX_LENGTH=256
prompt='''Given a query A and a passage B, determine whether the passage contains possible intents with the query by providing a prediction of either 'Yes' or 'No'.'''
def init_peft_ranker(pretrain_path,peft_path,use_fp16=True):
    ranker=FlagLLMReranker(pretrain_path,use_fp16=use_fp16)
    #import pdb;pdb.set_trace()
    if peft_path:
        print("loading peft params:",peft_path)
        ranker.model=PeftModel.from_pretrained(ranker.model,peft_path)
    return ranker
def init_ranker(pretrain_path):
    ranker=FlagReranker(pretrain_path)
    return ranker
def read_pickle(path):
    with open(path,'rb') as f:
        data=pickle.load(f)
    return data
def get_qid_results_cv(indri_rank_dict,ranker,test_qids):
    run_list=[]
    for qid in tqdm(test_qids):
        if qid=='95' or qid=='100':
            continue
        query,doc_list=indri_rank_dict[qid]
        docids=[item[0] for item in doc_list]
        doc_contents=[item[1][:MAX_LENGTH] for item in doc_list]
        rank_list=[[query,doc] for doc in doc_contents]
        #import pdb;pdb.set_trace()
        if isinstance(ranker,FlagLLMReranker):
            scores=ranker.compute_score(rank_list,batch_size=1,prompt=prompt)
        else:
            scores=ranker.compute_score(rank_list,batch_size=32)
        for i in range(len(docids)):
            docid=docids[i]
            score=scores[i]
            eval_item=ScoredDoc(qid,docid,score)
            run_list.append(eval_item)
    dict_result=pyndeval.ndeval(qrels, run_list, measures=["alpha-nDCG@20"])
    return dict_result
if __name__=="__main__":
    indri_rank_dict=read_pickle(indri_data)
    qrels=read_pickle(qrels_data)
    qrels=[SubtopicQrel(item[0],item[1],item[2],item[3]) for item in qrels]
    fold_id=[str(i+1) for i in range(5)]
    eval_metrics_overall=[]
    for fold in fold_id:
        peft_path_fold=peft_path+fold
        if not os.path.exists(peft_path_fold):
            peft_path_fold=None
        if "base" not in pretrain_path and "large" not in pretrain_path:
            ranker=init_peft_ranker(pretrain_path,peft_path_fold)
        else:
            ranker=init_ranker(pretrain_path)
        test_qids_path=cv_path+f"test_qids_fold{fold}.pkl"
        test_qids=read_pickle(test_qids_path)
        results_dict=get_qid_results_cv(indri_rank_dict,ranker,test_qids)
        eval_metrics=[results_dict[qid]['alpha-nDCG@20'] for qid in results_dict]
        eval_metrics_overall+=eval_metrics
    print("overall mean metrics:",np.mean(eval_metrics_overall))
    import pdb;pdb.set_trace()
    with open(save+".log",'w') as f:
        f.write(f"{save}, overall mean metrics:{np.mean(eval_metrics_overall)}\n")
    



from FlagEmbedding import FlagLLMReranker
from peft import PeftModel
import pickle
import pyndeval
from pyndeval import SubtopicQrel, ScoredDoc
from tqdm import tqdm
import numpy as np
indri_data='diversification_data/indri_data_installed.pkl'
qrels_data='diversification_data/qrels_installed.pkl'
#ranker_path='./merged_gemma-2b'
#ranker_path='./bge-reranker-v2-gemma'
pretrain_path='./gemma-2b'
peft_path='save_path/init_selftrain'
MAX_LENGTH=512
prompt='''Given a query A and a passage B, determine whether the passage contains possible intents with the query by providing a prediction of either 'Yes' or 'No'.'''
def init_peft_ranker(pretrain_path,peft_path,use_fp16=True):
    ranker=FlagLLMReranker(pretrain_path,use_fp16=use_fp16)
    ranker.model=PeftModel.from_pretrained(ranker.model,peft_path)
    return ranker
def read_pickle(path):
    with open(path,'rb') as f:
        data=pickle.load(f)
    return data
if __name__=="__main__":
    indri_rank_dict=read_pickle(indri_data)
    qrels=read_pickle(qrels_data)
    qrels=[SubtopicQrel(item[0],item[1],item[2],item[3]) for item in qrels]
    ranker=init_peft_ranker(pretrain_path,peft_path)
    run_list=[]
    for qid in tqdm(indri_rank_dict):
        if qid=='95' or qid=='100':
            continue
        query,doc_list=indri_rank_dict[qid]
        docids=[item[0] for item in doc_list]
        doc_contents=[item[1][:MAX_LENGTH] for item in doc_list]
        rank_list=[[query,doc] for doc in doc_contents]
        scores=ranker.compute_score(rank_list,batch_size=8,prompt=prompt)
        for i in range(len(docids)):
            docid=docids[i]
            score=scores[i]
            eval_item=ScoredDoc(qid,docid,score)
            run_list.append(eval_item)
    dict_result=pyndeval.ndeval(qrels, run_list, measures=["alpha-nDCG@20"])

    eval_metrics=[dict_result[qid]['alpha-nDCG@20'] for qid in dict_result]
    print(np.mean(eval_metrics))
    



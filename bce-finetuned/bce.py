import transformers
from functools import cache

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


@cache
def load_bce_embedding(local_files_only=True, device='cuda'):
    print('[bce.py] loading `bce-embedding-base_v1` model...')
    tokenizer = AutoTokenizer.from_pretrained(
        'maidalun1020/bce-embedding-base_v1', local_files_only=local_files_only)
    model: transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaPreTrainedModel = AutoModel.from_pretrained(
        'maidalun1020/bce-embedding-base_v1', local_files_only=local_files_only)
    model.to(device)
    return tokenizer, model



@cache
def load_bce_rerank(local_files_only=True, device='cuda'):
    print('[bce.py] loading `bce-reranker-base_v1` model...')
    model: transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
        'maidalun1020/bce-reranker-base_v1', local_files_only=local_files_only
    )
    tokenizer: transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast = AutoTokenizer.from_pretrained(
        'maidalun1020/bce-reranker-base_v1', local_files_only=local_files_only
    )
    model.to(device)
    return tokenizer, model


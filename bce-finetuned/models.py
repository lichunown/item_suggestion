import transformers
from functools import cache

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaClassificationHead
from bce import load_bce_embedding


class PairwiseClassificationModel(nn.Module):

    def __init__(self):
        super().__init__()
        _, pretrained_model = load_bce_embedding()
        self.embeddings = pretrained_model
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768, 1)

    def forward(self, **x):

        outputs = self.embeddings(**x, return_dict=True)
        x = outputs.last_hidden_state[:, 0]
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


if __name__ == '__main__':
    model = PairwiseClassificationModel().to('cuda')
    tokenizer, _ = load_bce_embedding()

    inputs = tokenizer(['a>B'], padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs_on_device = {k: v.to('cuda') for k, v in inputs.items()}

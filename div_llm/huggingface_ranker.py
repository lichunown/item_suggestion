import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )
pretrain_path=f"/root/autodl-tmp/pretrained_models/qwen1.5-1.8b-chat"
tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
if tokenizer.pad_token_id is None:
    if tokenizer.unk_token_id is not None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eod_id is not None:
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.im_start_id
        tokenizer.eos_token_id = tokenizer.im_end_id
if 'mistral' in pretrain_path.lower():
    tokenizer.padding_side = 'left'
import pdb;pdb.set_trace()
model = AutoModelForCausalLM.from_pretrained(pretrain_path)
yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = get_inputs(pairs, tokenizer)
    scores = model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float()
    print(scores)
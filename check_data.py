import json
with open("finetune_data_bge.jsonl") as f:
    lines=[line.strip() for line in f.readlines()]
count=0
for line in lines:
    dict_in=json.loads(line)
    if len(dict_in["neg"])==0:
        count+=1
print(count)
print(len(lines))
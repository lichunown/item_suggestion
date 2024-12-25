PRETRAIN_PATH=/root/autodl-tmp/pretrained_models/$1
SAVE_PATH=save_path/$2
DATA_PATH=$3
if [ ! -d "$SAVE_PATH" ]; then
    # 路径不存在，创建目录
    mkdir -p "$SAVE_PATH"
fi
#sh run_finetune_base.sh $SAVE_PATH/fold1 diversification_data/cross_validation/$3/train_jsonl_fold1.jsonl $PRETRAIN_PATH
#sh run_finetune_base.sh $SAVE_PATH/fold2 diversification_data/cross_validation/$3/train_jsonl_fold2.jsonl $PRETRAIN_PATH
#sh run_finetune_base.sh $SAVE_PATH/fold3 diversification_data/cross_validation/$3/train_jsonl_fold3.jsonl $PRETRAIN_PATH
#sh run_finetune_base.sh $SAVE_PATH/fold4 diversification_data/cross_validation/$3/train_jsonl_fold4.jsonl $PRETRAIN_PATH
sh run_finetune_base.sh $SAVE_PATH/fold5 diversification_data/cross_validation/$3/train_jsonl_fold5.jsonl $PRETRAIN_PATH
python3.10 div_llm/div_reranker_cv_subtopic_top100.py $1 $2 $3
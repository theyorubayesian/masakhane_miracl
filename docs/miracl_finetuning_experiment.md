# Finetuning on Miracl 
This example demonstrates how we can continue finetuning the dense retriever we trained on MS Marco. See [MS Marco Finetuning Experiment](msmarco_finetuning_experiment.md). This time we finetune and evaluate on the Miracl dataset.

## Finetuning on Train Set

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=masakhane-miracl WANDB_ENTITY=masakhane-miracl \
python -m tevatron.driver.train \
--output_dir models/swahili-bert-msmarco-miracl/ \
--model_name_or_path models/swahili-bert-msmarco/ \
--save_steps 20000 \
--dataset_name miracl/miracl:sw \
--fp16 \
--per_device_train_batch_size 8 \
--train_n_passages 8 \
--learning_rate 5e-6 \
--q_max_len 16 \
--p_max_len 128 \
--num_train_epochs 5 \
--logging_steps 500 \
--overwrite_output_dir \
--report_to wandb
```

## Evaluation on Dev Set

You can follow the steps detailed in [evaluating_on_miracl_dev_set](evaluating_on_miracl_dev_set.md). Be sure to edit the `--model_name_or_path`, `--output_dir`, and `--encoded_save_path`.

## Encoding `testA` Queries
`testA` set contains the queries we are finding rankings for. 

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m tevatron.driver.encode \
--model_name_or_path models/swahili-bert-msmarco-miracl \
--output_dir outputs/swahili-bert-msmarco-miracl/query_encoding \
--fp16 \
--per_device_eval_batch_size 256 \
--dataset_name miracl/miracl:sw/testA \
--encoded_save_path outputs/swahili-bert-msmarco-miracl/query_encoding/swahili/miracl_testA_query_emb.pkl \
--q_max_len 64 \
--encode_is_qry
```

## Encoding Corpus

If you already performed evaluation on the Miracl `dev` set using the model from this experiment, you can safely skip this step.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m tevatron.driver.encode \
--output_dir outputs/swahili-bert-msmarco-miracl/corpus_encoding \
--model_name_or_path models/swahili-bert-msmarco-miracl \
--fp16 \
--per_device_eval_batch_size 256 \
--p_max_len 256 \
--dataset_name miracl/miracl-corpus:sw \
--encoded_save_path outputs/swahili-bert-msmarco-miracl/corpus_encoding/swahili/miracl_corpus_emb.pkl \
--encode_num_shard 1
```

## Searching

Create the rankings file to be submitted for evaluation on the leaderboard or fused with other rankings.

```bash
python -m tevatron.faiss_retriever \
--query_reps outputs/swahili-bert-msmarco-miracl/query_encoding/swahili/miracl_query_emb.pkl \
--passage_reps outputs/swahili-bert-msmarco/corpus_encoding/swahili/miracl_corpus_emb.pkl \
--depth 100 \
--batch_size -1 \
--save_text \
--save_ranking_to runs/swahili-bert-msmarco/swahili/miracl.test.txt && \
python -m tevatron.utils.format.convert_result_to_trec \
--input runs/swahili-bert-msmarco/swahili/miracl.test.txt \
--output runs/swahili-bert-msmarco/swahili/miracl.test.trec
```

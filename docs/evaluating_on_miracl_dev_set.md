# Evaluating a Trained Dense Retriever on the Miracl Dev Set

This example demonstrates how we can perform evaluation on the Miracl Swahili dev set. It uses the Dense Retriever we trained on the MS Marco Passage Ranking Dataset (see [MS Marco Finetuning Experiment](msmarco_finetuning_experiment.md)) so it is a zero-shot evaluation. 

* Download the dev qrels

```bash
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.miracl-v1.0-sw-dev.tsv -O data/qrels.miracl-v1.0-sw-dev.tsv
```

* You will need to sign in to HuggingFace Hub through the command line in order to use the gated `miracl/miracl` dataset.

```
huggingface-cli login
```

* Encode Query: Note that tevatron does not support multi-GPU encoding for now.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m tevatron.driver.encode \
--model_name_or_path models/swahili-bert-msmarco \
--output_dir outputs/swahili-bert-msmarco/query_encoding \
--fp16 \
--per_device_eval_batch_size 256 \
--dataset_name miracl/miracl:sw/dev \
--encoded_save_path outputs/swahili-bert-msmarco/query_encoding/swahili/miracl_query_emb.pkl \
--q_max_len 64 \
--encode_is_qry
```

* Encode Corpus

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m tevatron.driver.encode \
--output_dir outputs/swahili-bert-msmarco/corpus_encoding \
--model_name_or_path models/swahili-bert-msmarco \
--fp16 \
--per_device_eval_batch_size 256 \
--p_max_len 256 \
--dataset_name miracl/miracl-corpus:sw \
--encoded_save_path outputs/swahili-bert-msmarco/corpus_encoding/swahili/miracl_corpus_emb.pkl \
--encode_num_shard 1
```

## Search

```bash
python -m tevatron.faiss_retriever \
--query_reps outputs/swahili-bert-msmarco/query_encoding/swahili/miracl_query_emb.pkl \
--passage_reps outputs/swahili-bert-msmarco/corpus_encoding/swahili/miracl_corpus_emb.pkl \
--depth 100 \
--batch_size -1 \
--save_text \
--save_ranking_to runs/swahili-bert-msmarco/swahili/miracl.dev.txt && \
python -m tevatron.utils.format.convert_result_to_trec \
--input runs/swahili-bert-msmarco/swahili/miracl.dev.txt \
--output runs/swahili-bert-msmarco/swahili/miracl.dev.trec
```

## Evaluation

```
python -m pyserini.eval.trec_eval -c \
-mrecip_rank \
-mrecall.100 \
data/qrels.miracl-v1.0-sw-dev.tsv \
runs/swahili-bert-msmarco/swahili/miracl.dev.trec > \
runs/swahili-bert-msmarco/swahili/miracl.dev.results
```
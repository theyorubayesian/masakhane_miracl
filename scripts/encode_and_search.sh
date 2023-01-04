export CUDA_VISIBLE_DEVICES=2

# This script only runs the `Search` only if it successfully completed both the `Query & Passage Encoding``
# If any or both encodings have already completed, uncomment either or both or lines 5 & 6, & the relevant command(s)
# enc_query=true
# enc_corpus=true

# This script only runs evaluation if search has been completed.
# Uncomment the line below and the commands for encoding and search to evaluate only.
# faiss_search=true

# Query Encoding
python -m tevatron.driver.encode \
--output_dir=outputs/davlan-swahili-bert/encoded_queries \
--model_name_or_path models/davlan-swahili-bert \
--fp16 \
--per_device_eval_batch_size 256 \
--dataset_name castorini/mr-tydi:swahili/test \
--encoded_save_path outputs/davlan-swahili-bert/query_embedding/swahili/query_emb.pkl \
--q_max_len 64 \
--encode_is_qry && enc_query=true

# Corpus Encoding
python -m tevatron.driver.encode \
--output_dir=outputs/davlan-swahili-bert/encoded_corpus \
--model_name_or_path models/davlan-swahili-bert \
--fp16 \
--per_device_eval_batch_size 256 \
--p_max_len 256 \
--dataset_name castorini/mr-tydi-corpus:swahili \
--encoded_save_path outputs/davlan-swahili-bert/corpus_encoding/swahili/corpus_emb.pkl \
--encode_num_shard 1 && enc_corpus=true

# Search
if [[ $enc_corpus && $enc_query ]]; then
    python -m tevatron.faiss_retriever \
    --query_reps outputs/davlan-swahili-bert/query_embedding/swahili/query_emb.pkl \
    --passage_reps outputs/davlan-swahili-bert/corpus_encoding/swahili/corpus_emb.pkl \
    --depth 100 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to runs/davlan-swahili-bert/swahili/mrtydi.test.txt && \
    python -m tevatron.utils.format.convert_result_to_trec \
    --input runs/davlan-swahili-bert/swahili/mrtydi.test.txt \
    --output runs/davlan-swahili-bert/swahili/mrtydi.test.trec && \
    faiss_search=true
fi

# Evaluation
if [[ $faiss_search ]]; then
  python -m pyserini.eval.trec_eval -c \
  -mrecip_rank \
  -mrecall.100 \
  /store2/scratch/aooladip/mr-tydi/mrtydi-v1.0-swahili/qrels.test.txt \
  runs/davlan-swahili-bert/swahili/mrtydi.test.trec > runs/davlan-swahili-bert/swahili/mrtydi.test.results
fi
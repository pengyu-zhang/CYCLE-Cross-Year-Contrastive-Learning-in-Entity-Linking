# CYCLE-Cross-Year-Contrastive-Learning-in-Entity-Linking

The implementation of our approach is based on the original codebase [BLINK](https://github.com/facebookresearch/BLINK) and [HeCo](https://github.com/liun-online/HeCo).<br>

<br><br>
<div align="center">
<img src="fig.png" width="800" />
</div>
<br><br>

Knowledge graphs constantly evolve with new entities emerging, existing definitions being revised, and entity relationships changing. These changes lead to temporal degradation in entity linking models, characterized as a decline in model performance over time. To address this issue, we propose leveraging graph relationships to aggregate information from neighboring entities across different time periods. This approach enhances the ability to distinguish similar entities over time, thereby minimizing the impact of temporal degradation. We introduce CYCLE: Cross-Year Contrastive Learning for Entity-Linking. This model employs a novel graph contrastive learning method to tackle temporal performance degradation in entity linking tasks. Our contrastive learning method treats newly added graph relationships as positive samples and newly removed ones as negative samples. This approach helps our model effectively prevent temporal degradation, achieving a 13.90% performance improvement over the state-of-the-art model from 2023 when the time gap is one year, and a 17.79% improvement as the gap expands to three years. Further analysis shows that CYCLE is particularly robust for low-degree entities, which are less resistant to temporal degradation due to their sparse connectivity, making them particularly suitable for our method.

## Usage

Please follow the instructions next to reproduce our experiments, and to train a model with your own data.

### 1. Install the requirements

Creating a new environment (e.g. with `conda`) is recommended. Use `requirements.txt` to install the dependencies:

```
conda create -n cycle311 -y python=3.11 && conda activate gclel311
pip install -r requirements.txt
```

### 2. Download the data

| Download link                                                | Size |
| ------------------------------------------------------------ | ----------------- |
| [Our Dataset](https://zenodo.org/records/10977757) | 3.12 GB            |
| [ZESHEL](https://github.com/facebookresearch/BLINK/tree/main/examples/zeshel) | 1.55 GB            |
| [WikiLinksNED](https://github.com/yasumasaonoe/ET4EL) | 1.1 GB             |

### 3. Reproduce the experiments

```
export PYTHONPATH="/code/cycle:$PYTHONPATH"
python /code/gcl/blink/biencoder/train_biencoder.py \
 --data_path /code/dataset/01_blink_baseline/blink_format/mix_1764/2013/ \
 --output_path /models/mix_1764_gcl/2013/biencoder \
 --learning_rate 1e-05 --num_train_epochs 1 --weight 45 --max_context_length 128 --max_cand_length 128 \
 --train_batch_size 16 --eval_batch_size 16 --bert_model google/bert_uncased_L-8_H-512_A-8 \
 --type_optimization all_encoder_layers --data_parallel \

# Get top-64 predictions from Biencoder model on train, valid and test dataset
export PYTHONPATH="/code/blink:$PYTHONPATH"
python /code/blink/blink/biencoder/eval_biencoder.py \
 --path_to_model /models/mix_1764_gcl/2013/biencoder/pytorch_model.bin \
 --data_path /code/dataset/01_blink_baseline/blink_format/mix_1764/ \
 --output_path /models/mix_1764_gcl/2013/ \
 --encode_batch_size 32 --eval_batch_size 32 --top_k 64 --save_topk_result \
 --bert_model google/bert_uncased_L-8_H-512_A-8 --mode 2013,2014,2015,2016,2017,2018,2019,2020,2021,2022 \
 --zeshel True --data_parallel \
```
<br><br>
<div align="center">
<img src="fig2.png" width="700" />
</div>
<br><br>

## Using your own data

If you want to use your own dataset, you only need to use the code in Dataset Construction. Construct your own dataset according to the description of the dataset construction process in the Supplementary Material.

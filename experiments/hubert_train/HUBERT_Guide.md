# HuBERT

4 GPUs, Nvidia V100 32GB, precision fp32.

```shell

cd /home/vrdauer/VK-SSL/experiments/hubert_train
```

## 0. Datasets and duration caches

`/home/vrdauer/VK-SSL/experiments/ctc_train/librispeech`

`LibriSpeech: train-500, traint-360, train-clean-100,dev-clean,dev-other`

`Libri-Light: librispeech_finetuning`


```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m src.data.duration_cache \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --subset 100h ll-10h \
        --scan-workers 32
```

## 1. Teacher labels, iteration 1

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python extract_features.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --out-dir /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h \
        --feature-type mfcc \
        --subsets train-clean-100 dev-clean dev-other
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python learn_kmeans.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/km_100.bin \
        --n-clusters 100 \
        --percent 1.0 \
        --max-frames 50000000 \
        --batch-size 10000 \
        --n-init 20 \
        --max-iter 100
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python dump_labels.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/km_100.bin \
        --out-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/all.km
```

## 2. Pre-train on 100 hours

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m torch.distributed.run --nproc_per_node=4 train_hubert.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --label-paths /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/all.km \
        --exp-dir /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1_100h \
        --train-subsets train-clean-100 \
        --model-size base \
        --num-classes 100 \
        --label-rate 100 \
        --mask-alpha 1.0 \
        --mask-prob 0.08 \
        --lr 5e-4 \
        --weight-decay 0.01 \
        --warmup-ratio 0.08 \
        --max-batch-duration 87.5 \
        --max-steps 250000 \
        --accumulate-grad-batches 8 \
        --gpus 4
```

## 3. Iteration 2

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python extract_features.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --out-dir /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert6_100h \
        --feature-type hubert \
        --checkpoint-path /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1_100h/checkpoints/<best>.ckpt \
        --model-size base \
        --num-classes 100 \
        --label-rate 100 \
        --layer 6 \
        --use-cuda \
        --subsets train-clean-100 dev-clean dev-other
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python learn_kmeans.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert6_100h/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert6_100h/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert6_100h/km_500.bin \
        --n-clusters 500 \
        --percent 0.1 \
        --batch-size 10000 \
        --n-init 20 \
        --max-iter 100
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python dump_labels.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert6_100h/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert6_100h/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert6_100h/km_500.bin \
        --out-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert6_100h/all.km
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m torch.distributed.run --nproc_per_node=4 train_hubert.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --label-paths /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert6_100h/all.km \
        --exp-dir /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it2_100h \
        --train-subsets train-clean-100 \
        --model-size base \
        --num-classes 500 \
        --label-rate 50 \
        --mask-alpha 1.0 \
        --mask-prob 0.08 \
        --lr 5e-4 \
        --weight-decay 0.01 \
        --warmup-ratio 0.08 \
        --max-batch-duration 87.5 \
        --max-steps 400000 \
        --accumulate-grad-batches 8 \
        --gpus 4
```

## 4. Fine-tune on 10 hours

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m torch.distributed.run --nproc_per_node=4 finetune_hubert.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --pretrained-path /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1_100h/checkpoints/<best>.ckpt \
        --exp-dir /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_ft10h \
        --preset 10h \
        --model-size base \
        --num-classes 100 \
        --label-rate 100 \
        --gpus 4
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python eval_hubert.py \
        --checkpoint-path /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_ft10h/checkpoints/<best>.ckpt \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --label-type char \
        --use-cuda
```

## Reference

### Pre-train settings vs fairseq

| Item | fairseq | Repo |
|---|---|---|
| Transformer | 12 / 768 / **12 heads** / 3072 | same |
| mask | `mask_prob 0.80` = 0.08·T starts, span 10, `min_masks 2` | `--mask-prob`, our formula counts starts directly, so fairseq value ÷ 10 |
| loss | masked-only + `10 · mean(conv_out²)` | `mask_alpha=1.0`, `feature_penalty_weight=10.0` |
| `feature_grad_mult` | 0.1 | same |
| `clip_norm` | 10.0 | `gradient_clip_val=10.0` |
| Adam | lr `5e-4`, β `(0.9,0.98)`, eps `1e-6`, wd `0.01` (**decoupled**, fairseq Adam ≡ AdamW) | `torch.optim.AdamW`, same values |
| schedule | `polynomial_decay`, warmup **8%**, decay to 0 | `warmup_ratio=0.08` |
| steps | it1 **250k**, it2 **400k** (960 h) | `--max-steps` |
| batch | 32 GPUs × 87.5 s = 2800 s | `--gpus × --max-batch-duration × --accumulate-grad-batches` |
| precision | fp16 | **fp32 hardcoded** |
| labels | it1 MFCC → 100 @ 100 Hz, it2 layer 6 → 500 @ 50 Hz | `--num-classes`, `--label-rate` |

### Fine-tune presets

| Preset | data | lr | steps | tri-stage warmup/hold/decay | mask | effective batch |
|---|---|---|---|---|---|---|
| `10h` | Libri-Light 10h (`ll-10h`) | 2e-5 | 25000 | 8000 / 0 / 72000 | 0.075 | 200 s |
| `100h` | `train-clean-100` | 3e-5 | 80000 | 8000 / 32000 / 40000 | 0.065 | 1600 s |

| Item | fairseq | Repo |
|---|---|---|
| targets | `labels: ["ltr"]` — 26 letters + `'` + `\|` | `--label-type char`, 28 + blank |
| criterion | ctc, `zero_infinity`, `sentence_avg` | same |
| optimizer | Adam, β `(0.9,0.98)`, eps **1e-8**, no wd | same |
| `final_lr_scale` | 0.05 | same |
| freeze | `freeze_finetune_updates 10000` | `--freeze-steps 10000` |
| channel mask | `mask_channel_prob 0.5`, length 64 | same |
| dropouts | `dropout 0`, `attention 0`, `activation 0.1`, `layerdrop 0.1` | same |
| `feature_grad_mult` | 0.0 | CNN frozen |

## Full 960 hours

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m src.data.duration_cache \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --subset all \
        --scan-workers 32
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python extract_features.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --out-dir /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc \
        --feature-type mfcc \
        --subsets train-clean-100 train-clean-360 train-other-500 dev-clean dev-other
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python learn_kmeans.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/km_100.bin \
        --n-clusters 100 \
        --percent 1.0 \
        --max-frames 50000000 \
        --batch-size 10000 \
        --n-init 20 \
        --max-iter 100
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python dump_labels.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/km_100.bin \
        --out-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/all.km
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m torch.distributed.run --nproc_per_node=4 train_hubert.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --label-paths /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/all.km \
        --exp-dir /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1 \
        --model-size base \
        --num-classes 100 \
        --label-rate 100 \
        --mask-alpha 1.0 \
        --mask-prob 0.08 \
        --lr 5e-4 \
        --weight-decay 0.01 \
        --warmup-ratio 0.08 \
        --max-batch-duration 87.5 \
        --max-steps 250000 \
        --accumulate-grad-batches 8 \
        --gpus 4
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python compute_ger.py \
        --checkpoint-path /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1_100h/checkpoints/<best>.ckpt \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --model-size base \
        --num-classes 100 \
        --label-rate 100 \
        --layer 12 \
        --subsets train-clean-100 \
        --max-seconds 3600 \
        --use-cuda \
        --out-json /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1_100h/ger.json
```
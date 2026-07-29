# VK-SSL

## Окружение

### Локальное окружение
```shell

git checkout preparing_backbone

python3 -m venv .venv
source .venv/bin/activate

pip install -e
pip install -r requirements.txt
pip uninstall torchvision -y
```

### Окружение суперкомпьютера

```shell

module load Python/PyTorch_GPU_v2.4
source ~/vk-ssl-venv/bin/activate
```

## Dev-clean
```shell

cd experiments/ctc_train 
python3 download_dev_subset_libri.py
```

## Токенизатор
```shell

cd experiments/ctc_train
python3 train_spm.py \
    --librispeech-path ./librispeech \
    --output-file ./librispeech/spm_unigram_1023.model
```

## Обучение
```shell

cd experiments/ctc_train 
PYTHONPATH=/home/vrdauer/VK-SSL python -m torch.distributed.run \
    --nproc_per_node=4 train_ctc.py \
    --exp-dir ./librispeech/logs \
    --librispeech-path ./librispeech \
    --global-stats-path ./global_stats.json \
    --sp-model-path ./librispeech/spm_unigram_1023.model \
    --epochs 150 \
    --gpus 4
```

## WER
```shell

cd experiments/ctc_train 
PYTHONPATH=/home/vrdauer/VK-SSL python3 -W ignore eval_ctc.py \
    --checkpoint-path ./librispeech/logs/checkpoints/'checkpoint_name'.ckpt \
    --librispeech-path ./librispeech \
    --sp-model-path ./librispeech/spm_unigram_1023.model \
    --global-stats-path ./global_stats.json \
    --use-cuda 
```

## TenserBoard
```shell

cd experiments/ctc_train 
tensorboard --logdir=./librispeech/logs/lightning_logs
```

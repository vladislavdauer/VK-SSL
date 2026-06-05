# VK-SSL

## Окружение
```shell

git checkout preparing_backbone

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e
pip install torchaudio==2.4.0
pip uninstall torchvision -y
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
PYTHONPATH=/home/vladislavdauer/PycharmProjects/VK-SSL python3 train_ctc.py \
    --exp-dir ./librispeech/logs \
    --librispeech-path ./librispeech \
    --global-stats-path ./global_stats.json \
    --sp-model-path ./librispeech/spm_unigram_1023.model \
    --sanity_check \
    --epochs 100
```

## TenserBoard
```shell

cd experiments/ctc_train 
tensorboard --logdir=./librispeech/logs/lightning_logs
```

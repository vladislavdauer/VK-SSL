# VK-SSL

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
<!-- deactivate -->

<!-- Скачать dev-clean для sanity check -->
cd experiments/rnnt_train 
python3 download_dev_subset_libri.py
<!-- Ну или сказать через впн и отдельно положить в *tar.gz -->
tar -xzf librispeech/dev-clean.tar.gz -C librispeech

<!-- Столкнулся с проблемами использования ffmpeg в torchaudio на mac -->
<!-- Либо понизить версию pip install torchaudio==2.4.0 -->
<!-- Тут либо можно через docker попробовать, команда будет выглядеть примерно так -->
docker run --rm -it \
  -v $(pwd)/experiments/rnnt_train/librispeech:/app/experiments/rnnt_train/librispeech \
  librispeech-sanity

<!-- Обучим токенизатор (например во время сейнити) -->
python train_spm.py --librispeech-path ./datasets --sanity_check

<!-- Проводим sanity -->
PYTORCH_ENABLE_MPS_FALLBACK=1 rnnt_train % python3 train_rnnt.py --exp-dir ./librispeech/logs/ --librispeech-path ./librispeech/ --global-stats-path ./global_stats.json --sp-model-path ./librispeech/spm_unigram_1023.model --epochs 1 --sanity_check

<!-- Смотрим логи в tb -->
tensorboard --logdir=./librispeech/logs/lightning_logs

<!-- Получаем чекпоинт на выходе -->
VK-SSL/experiments/rnnt_train/librispeech/logs/checkpoints/epoch=0-step=50-v1.ckpt
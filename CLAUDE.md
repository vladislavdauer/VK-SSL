# VK-SSL: CTC Speech Recognition на LibriSpeech

## Обзор проекта

Проект для обучения CTC (Connectionist Temporal Classification) модели распознавания речи на датасете LibriSpeech. Используется архитектура Conformer с SentencePiece токенизацией.

**Проблема:** На тесте WER ~30% вместо ожидаемых ~5% из статей.

---

## Структура репозитория

```
VK-SSL/
├── src/
│   ├── models/
│   │   ├── asr_lightning_module.py   # Lightning модуль для CTC и RNNT
│   │   ├── build_model.py            # Фабрика моделей (conformer_v2_ctc_base)
│   │   ├── conformer.py              # Оригинальный Conformer из torchaudio
│   │   ├── conformer_v2.py           # ConformerV2 из GigaAM (rotary embeddings)
│   │   ├── rnnt.py                   # RNN-T модель
│   │   └── rnnt_decoder.py           # Beam search decoder для RNNT
│   ├── data/
│   │   ├── librispeech_data_module.py # DataModule с bucket sampling
│   │   └── data_transforms.py        # Мел-спектрограммы, SpecAugment
│   └── opt/
│       └── schedulers.py             # NoamAnnealing, WarmupCosineScheduler
├── experiments/
│   └── ctc_train/
│       ├── train_ctc.py              # Скрипт обучения
│       ├── eval_ctc.py               # Оценка WER
│       ├── inference.py              # Инференс с beam search
│       ├── train_spm.py              # Обучение SentencePiece
│       ├── download_dev_subset_libri.py
│       └── global_stats.json         # Статистики для нормализации
└── requirements.txt
```

---

## Архитектура модели

### ConformerV2 CTC Base (conformer_v2_ctc_base)

```python
input_dim=80               # Мел-спектрограмма
encoding_dim=256           # Выходное измерение
time_reduction_stride=4    # Редукция времени 4x
conformer_input_dim=256
conformer_ffn_dim=1024     # 4x expansion
conformer_num_layers=16
conformer_num_heads=4
conformer_depthwise_conv_kernel_size=31
conformer_dropout=0.1
```

### Поток данных

1. **Input:** `(B, T, 80)` - мел-спектрограмма
2. **TimeReduction (stride=4):** `(B, T//4, 320)` - конкатенация 4 фреймов
3. **Input Linear:** `(B, T//4, 256)`
4. **ConformerV2 Encoder:** 16 слоев с rotary positional embeddings
   - Subsampling внутри отключена (`pre_encode = None`)
   - Self-attention: rotary, 4 heads
   - Conv kernel: 31
5. **Output Linear + LayerNorm:** `(B, T//4, 256)`
6. **CTC Head:** Linear(256, 1024) для 1023 токенов + 1 blank

### SentencePiece токенизация

- Vocab size: 1023
- Model type: unigram
- Special tokens: bos=0, pad=1, eos=2, unk=3
- Blank index: 1023 (последний)

---

## Подготовка данных

### 1. Скачивание dev-clean
```bash
cd experiments/ctc_train
python3 download_dev_subset_libri.py
```

### 2. Обучение SentencePiece
```bash
cd experiments/ctc_train
python3 train_spm.py \
    --librispeech-path ./librispeech \
    --output-file ./librispeech/spm_unigram_1023.model
```

### 3. Глобальная статистика
Файл `global_stats.json` содержит:
- `mean`: среднее по каждому мел-каналу (80 значений)
- `invstddev`: обратное стандартное отклонение

**Важно:** Статистики должны быть вычислены на обучающей выборке (train-clean-100 + train-clean-360 + train-other-500).

---

## Обучение

### Команда обучения
```bash
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

### Гиперпараметры обучения

| Параметр | Значение | Примечание |
|----------|----------|------------|
| Optimizer | AdamW | betas=(0.9, 0.98), weight_decay=1e-3 |
| Learning rate | 5.0 | Очень высокий стартовый LR |
| LR Scheduler | NoamAnnealing | d_model=256, warmup_steps=10000, min_lr=1e-6 |
| Gradient clip | 0.5 | Может быть агрессивным |
| Accumulate grad batches | 16 | Эффективный batch size ×16 |
| Max tokens per batch | 32000 | Для bucketing |
| Batch size | 32 | Базовый размер |
| Epochs | 150 | |
| Dropout | 0.1 | В конформере |

### Data Augmentation (SpecAugment)
- Frequency masking: 2x с width=27
- Time masking: 2x с width=100, p=0.2

---

## Оценка WER

### Команда оценки
```bash
cd experiments/ctc_train
PYTHONPATH=/home/vrdauer/VK-SSL python3 -W ignore eval_ctc.py \
    --checkpoint-path ./librispeech/logs/checkpoints/checkpoint_name.ckpt \
    --librispeech-path ./librispeech \
    --sp-model-path ./librispeech/spm_unigram_1023.model \
    --global-stats-path ./global_stats.json \
    --use-cuda \
    --subsets test-clean test-other
```

### Декодирование
- **Greedy decoding** в eval_ctc.py
- **Beam search** доступен в inference.py (требует JIT экспорта)

---

## Потенциальные проблемы (TODO для анализа)

### 1. Архитектура и поток данных

**Проблема: Двойная редукция времени**
- TimeReduction в wrapper: stride=4 (4x редукция)
- Subsampling в ConformerV2: factor=2 (2x редукция) 
- **Итого:** 8x редукция времени

**Файл:** `src/models/build_model.py:51`
```python
self.encoder.pre_encode = None  # Отключено, но subsampling_factor=2 в конфиге
```

**Вопрос:** Правильно ли вычисляются output lengths после всех редукций?

### 2. Learning Rate и оптимизация

**Проблема:** Стартовый LR=5.0 очень высокий для AdamW.

**Файл:** `src/models/asr_lightning_module.py:48-54`
```python
self.optimizer = torch.optim.AdamW(
    ...,
    lr=5.0,  # Очень высокий!
    eps=1e-9,
    betas=(0.9, 0.98),
    weight_decay=1e-3
)
```

В стандартных Conformer CTC используют LR ~0.001-0.002 с warmup.

### 3. Gradient Clipping

**Проблема:** clip_val=0.5 может быть слишком агрессивным.

**Файл:** `experiments/ctc_train/train_ctc.py:55`
```python
gradient_clip_val=0.5
```

### 4. Накопление градиентов

**Проблема:** accumulate_grad_batches=16 создает очень большой эффективный batch.

**Файл:** `experiments/ctc_train/train_ctc.py:58`
```python
accumulate_grad_batches=16
```

При batch_size=32 и 4 GPU: эффективный batch = 32 × 4 × 16 = 2048

### 5. Маскирование паддинга в ConformerV2

**Проблема:** Неправильная обработка паддинга в self-attention.

**Файл:** `src/models/build_model.py:66-116`
```python
# pad_mask создается вручную
pad_mask = (
    torch.arange(x.size(1), device=x.device).unsqueeze(0)
    >= lengths.unsqueeze(1)
)
x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
```

В отличие от оригинального Conformer, здесь используется ручное маскирование вместо key_padding_mask в MultiheadAttention.

### 6. Warmup steps

**Проблема:** warmup_steps=10000 может быть недостаточно.

**Файл:** `src/models/asr_lightning_module.py:173-178`
```python
self.warmup_lr_scheduler = NoamAnnealing(
    self.optimizer,
    d_model=256,
    warmup_steps=10000,  # Может быть мало для стабильного обучения
    min_lr=1e-6,
)
```

### 7. Проверка global_stats.json

**Вопрос:** Правильно ли вычислены mean и invstddev?

Формула в `data_transforms.py`:
```python
(input - mean) * invstddev  # Нормализация
```

**Важно:** invstddev должен быть 1/std, а не std.

### 8. Inference vs Eval mismatch

**Проблема:** Два разных способа инференса:

- `eval_ctc.py`: Загружает checkpoint напрямую через Lightning
- `inference.py`: Ожидает JIT-экспортированную модель

**Вопрос:** Оценивается ли WER правильно в eval_ctc.py?

### 9. Сравнение с эталонной архитектурой

Стандартный Conformer CTC (ESPnet, torchaudio) использует:
- input_dim=80
- encoder_dim=256
- num_layers=12 (не 16)
- num_heads=4
- conv_kernel_size=31
- subsampling=conv2d (4x)
- LR=0.002 с warmup

Отличия в текущей реализации:
- 16 слоев вместо 12
- Двойная редукция (4x time + 2x conv = 8x)
- Очень высокий LR=5.0
- Другая схема warmup

### 10. Проверка CTC Loss

**Файл:** `src/models/asr_lightning_module.py:46`
```python
self.loss = torch.nn.CTCLoss(blank=self.blank_idx, reduction="none", zero_infinity=True)
```

blank_idx=1023 (последний), что корректно.

---

## Рекомендации для исправления

1. **Проверить learning rate:** Уменьшить с 5.0 до 0.001-0.002
2. **Проверить warmup:** Увеличить до 25000-50000 шагов
3. **Проверить gradient clip:** Попробовать 1.0 или 5.0, или отключить
4. **Проверить accumulate_grad_batches:** Уменьшить до 1-4
5. **Проверить lengths:** Убедиться что длины правильно уменьшаются после редукций
6. **Проверить маскирование:** Убедиться что паддинг не влияет на attention
7. **Проверить global_stats:** Убедиться что mean/invstddev правильные
8. **Сравнить с бейзлайном:** Обучить на меньшем подмножестве с известными параметрами

---

## Полезные команды

### TensorBoard
```bash
cd experiments/ctc_train
tensorboard --logdir=./librispeech/logs/lightning_logs
```

### Sanity check
```bash
python3 train_ctc.py --sanity_check --gpus 1
```

### Проверка датасета
```python
from src.data.librispeech_data_module import get_data_module
dm = get_data_module("./librispeech", "./global_stats.json", "./spm_unigram_1023.model")
print(len(dm.train_dataloader()))
```

---

## История изменений (Git)

- d8085b2 Some style fix
- a9f2d53 Adding test-other subset
- b21f74e Fix test counter of WER
- d6eb551 Delete sanity.py
- 43e0a4b Try other scheduler
- b0efeb0 Dropout fix + new LR test
- 8050e2e Last train settings
- dc54866 New experiment params, logging val WER and grads

---

## Зависимости

- torch==2.4.0
- torchaudio==2.4.0
- pytorch-lightning
- sentencepiece
- tensorboard

---

## Контакты и вопросы

Для отладки WER 30% vs 5%:
1. Сравнить learning rate schedule с эталонными реализациями
2. Проверить корректность обработки паддинга в ConformerV2
3. Валидировать правильность вычисления lengths через все слои
4. Проверить что global_stats.json соответствует обучающей выборке

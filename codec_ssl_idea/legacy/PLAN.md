# HuBERT + codec-таргеты: план экспериментов

## 1. Что взяли из статей

| Источник | Что полезно |
|---|---|
| EnCodecMAE (codacs.pdf) | Таргеты = RVQ-индексы замороженного кодека; кросс-энтропия по каждой кодбуке со своим весом γ_q; взвешивание masked/unmasked (α, β); self-training (k-means по слоям модели) как второй этап. На ASR (SUPERB) заметно хуже HuBERT (WER 12.4 vs 3.6 с LM) – вывод авторов: «ключ не только в данных, но и в таргетах». |
| Codec2Vec (2511.16639) | Кодек как **вход**, цели – reconstruction / iterative k-means / online EMA-кластеризация (DinoSR). Хранилище 60.4→3.6 ГБ (16.5×), обучение 830→356 GPU-ч (2.3×), PER 5.5 vs 5.4 у HuBERT. |
| 2607.26350 (Takizawa et al.) | Качество не зависит от языка обучения кодека, но сильно зависит от языка SSL-претрейна → один кодек на все языки без дообучения. (Прочитан только abstract.) |

**Важная поправка к идее «нет работ»:** близко уже есть Codec2Vec (кодек-вход + online-кластеризация), BEST-RQ (случайный квантизатор: тоже без диска и итераций), DinoSR (online-кластеризация EMA-учителем), EnCodecMAE (кодек-таргеты, но MAE и mel-вход). Свободная ниша: **классический HuBERT (waveform → CNN → Transformer, mask-token) с кодек-таргетами вместо k-means, измеренный как задача эффективности** (GPU-часы, диск, число итераций, стабильность), плюс абляции по уровням RVQ.

## 2. Гипотезы (для статьи про скорость/продуктивность)

- **H1.** Кодек-таргеты (DAC-16k, 50 Гц = ровно частота CNN HuBERT) дают WER/PER в пределах ±5% от HuBERT-iter2 при **1 итерации** вместо 2 и **0 ГБ** промежуточных фич.
- **H2.** Online-режим (кодек считает таргеты на GPU) убирает дисковый I/O; цена – доп. forward кодека (~несколько % FLOPs против 95M Transformer).
- **H3.** Грубые RVQ-уровни (q0–q1) содержат фонетику, поздние – акустику/просодию/спикера; веса γ_q ↓ и coarse-to-fine curriculum дают лучше ASR, чем равные веса (аналог результата EnCodecMAE по γ).
- **H4.** Меньше стабильности проблем: дисперсия по сидам ниже, чем у k-means iter1 (чувствителен к инициализации) и BEST-RQ.
- **H5 (из 2607.26350).** Кодек, обученный на английском, работает для другого языка SSL-претрейна без изменений – проверяемо на небольшом нераспространённом корпусе.

## 3. Матрица экспериментов (Base, LibriSpeech 960h, затем fine-tune 10h/100h CTC, WER dev-clean/other)

| # | Таргеты | Комментарий |
|---|---|---|
| E0 | HuBERT iter1 (MFCC k-means 100) | baseline (есть в репо) |
| E1 | HuBERT iter2 (k-means 500 по слою 6 E0) | baseline «сколько стоит весь пайплайн» |
| E2 | BEST-RQ (рандомный квантизатор) | обязательный честный baseline «без диска/итераций» (папка `bestrq_train` пуста) |
| E3 | DAC-16k, q0 only (K=1024), online | минимальный кодек-вариант |
| E4 | DAC-16k, Q=4, равные веса | `--codec-num-q 4` |
| E5 | DAC-16k, Q=4, веса 1,.5,.25,.125 | H3 |
| E6 | DAC-16k, Q=4, curriculum (+1 голова/10k шагов) | H3 |
| E7 | Q ∈ {1,2,4,8,12} | кривая «число уровней – WER» |
| E8 | offline-dump vs online | одинаковое качество, разные время/диск |
| E9 | E5 + self-training (k-means по слою E5, как в EnCodecMAE) | потолок качества |
| E10 | Другие кодеки: EnCodec-24k (75 Гц), Mimi (12.5 Гц) | подключаются через `src/codec/targets.py` |
| E11 | Мульти-язык: кодек EN, SSL на другом языке | H5 |

**Метрики:** WER/PER (fine-tune 10h и 100h), masked-acc и perplexity таргетов по каждой кодбуке (логируются), GPU-часы до фикс. качества, `Perf/audio_sec_per_sec`, пиковая память, ГБ на диске, сиды ×3.

**Ожидаемый результат (гипотезы, не факты):** E4–E6 – WER на 10h в пределах нескольких % отн. от E1 и лучше E2; выигрыш по времени ~2× относительно полного пайплайна E0+E1 (по аналогии с Codec2Vec 2.3×); основная деградация, если будет – на поздних слоях/ASR, как у EnCodecMAE, поэтому E9 нужен как страховка.

## 4. Что уже сделано в коде (ветка `codec-targets`, `codec_targets.patch`)

- `src/codec/targets.py` – `CodecTargetExtractor` (DAC-обёртка, ресемплинг индексов по времени для кодеков не 50 Гц, `StubCodec` для тестов).
- `hubert_lightning_module.py` – `--targets codec-online`, замороженный кодек вне state_dict/DDP; per-codebook masked-acc и target-perplexity; метрики throughput/память; coarse-to-fine curriculum.
- `hubert_model.py` / `config.py` – веса кодбук `codebook_weights` (γ_q), `active_heads`.
- `dump_codec_labels.py` – offline-таргеты в том же `.km` формате (без правок датапайплайна).
- **Ускорения (независимо от кодека, дают отдельный пункт в статье):**
  - `prediction_head.py`: косинус через нормировку + matmul вместо `F.cosine_similarity` на `[N,K,D]` (при K=1024, D=256 и 87.5 с аудио ≈ 4.3 ГБ на голову), логиты только на masked-кадрах;
  - `masking.py`: векторизованное span-маскирование без python-цикла по батчу.
- `--precision bf16-mixed` в `train_hubert.py`.

Запуск:

```
# online
python -m experiments.hubert_train.train_hubert --librispeech-path ... \
  --targets codec-online --codec dac-16khz --codec-num-q 4 \
  --codebook-weights 1,0.5,0.25,0.125 --label-rate 50 --precision bf16-mixed
# offline
python -m experiments.hubert_train.dump_codec_labels --librispeech-path ... --num-q 4 --out-dir codec_labels
python -m experiments.hubert_train.train_hubert ... --label-paths codec_labels/codec_q*.km --num-classes 1024,1024,1024,1024 --label-rate 50
```

## 5. Проверено / не проверено

Проверено на CPU (5 unit-тестов, `test_codec_targets.py`): эквивалентность логитов старой реализации, инварианты и доля маски (~57% как у прежнего цикла), мульти-голова + веса + curriculum, обучение падает на stub-кодеке.
**Не проверено:** `pytorch_lightning`/`torchaudio`/`dac` в окружении не установлены → `HubertPretrainModule`, `DACCodec`, `dump_codec_labels.py` и CLI проверены только компиляцией; не запускался реальный DAC; ускорение (время, память) не измерялось, цифры выше – расчёт; существующие тесты репо (`test_hubert.py`) не прогонялись (нужен Lightning).

## 6. Риски

1. Кодек-таргеты шумные по фонетике (акустика, спикер) → хуже ASR, чем k-means по HuBERT-слою (результат EnCodecMAE). Митигация: веса, curriculum, E9.
2. Новизна пересекается с Codec2Vec/BEST-RQ → позиционировать как «target study + efficiency», сравнивать напрямую с E2.
3. DAC-16k – 12 кодбук; 4 из них × K=1024 = 4 головы по 1024 → доп. память на логиты (уже снижена в коде).

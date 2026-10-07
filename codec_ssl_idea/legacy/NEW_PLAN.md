# HuBERT + codec-таргеты: рассуждения, план и обзор патча

## 1. Контекст

Идея: заменить k-means-таргеты HuBERT (MFCC k-means → layer-6 k-means) на RVQ-индексы замороженного нейрокодека (DAC-16k, EnCodec-24k). Цель — убрать итерации кластеризации и дисковый дамп фич, измерить выигрыш по GPU-часам, памяти и стабильности при сохранении WER/PER.

Позиционирование: это не «новый метод таргетов», а **systematic study of efficiency-quality trade-off** для SSL-претрейна речи.

## 2. Ключевое различие: HuBERT+codec vs BEST-RQ+codec

Если в BEST-RQ **полностью** заменить random quantizer (random projection + random codebook) на «энкодер кодека + RVQ-кодбуки», то **генерация таргетов становится идентичной** предложенной схеме: те же индексы, из того же кодека, с той же частотой.

Разница остаётся только в:
- **SSL-архитектуре**: BEST-RQ (mel + Conformer) vs HuBERT (waveform → CNN → Transformer).
- **Деталях обучения**: маскирование mel-кадров vs CNN-фич; своя prediction head; одноуровневый vs многоуровневый RVQ.
- **Стоимости таргетов**: BEST-RQ теряет своё главное преимущество (почти бесплатные таргеты), как только вы ставите кодек.

**Вывод для статьи:** «BEST-RQ + codec» — обязательный baseline. Без него вклад выглядит как переименование BEST-RQ. Новизна — не в таргетах, а в комбинации: HuBERT-архитектура + кодек-таргеты + multi-level RVQ + curriculum + измерение эффективности.

## 3. Выбор кодека

Для простоты и чистоты эксперимента:

- **DAC-16k** — оптимален. Нативно 50 Гц, совпадает с HuBERT CNN **без ресемплинга**. 12 RVQ-уровней по 1024. Можно брать только q0 для минимального варианта.
- **EnCodec 24k** — рабочий вариант, но 75 Гц → нужен ресемплинг индексов (75→50) или изменение stride CNN. Добавляет шаг предобработки и потенциальные артефакты выравнивания.
- **WavTokenizer** — одноуровневый, 40 Гц, кодбук 4096. Много логитов, ресемплинг 40→50 даёт неравномерное дублирование токенов. Не подходит.
- **SpeechTokenizer** — многоуровневый RVQ, но с семантической дистилляцией (q0 обучается с учителем HuBERT). Это «очищает» таргет и убивает именно тот эффект, который вы хотите измерить. Не подходит.

**Итог:** DAC-16k как основной, EnCodec 24k как дополнительный.

## 4. План экспериментов (v2, сжато)

**Базовые:**
- E0: HuBERT iter1 (MFCC k-means 100)
- E1: HuBERT iter2 (k-means 500 по слою 6)
- E2: BEST-RQ (random quantizer)
- E2c: BEST-RQ + codec q0 only (новый, обязательный)
- E2m: BEST-RQ + codec Q=4 + веса (новый)

**Основные:**
- E3: DAC-16k, q0 only, online
- E4: DAC-16k, Q=4, равные веса
- E5: DAC-16k, Q=4, веса 1, .5, .25, .125
- E6: DAC-16k, Q=4, curriculum (+1 голова/10k шагов)
- E7: Q ∈ {1,2,4,8,12}
- E8: offline-dump vs online
- E9: E5 + self-training (k-means по слою E5)
- E10: EnCodec-24k, Mimi
- E11: мульти-язык (H5)
- E12: Codec2Vec-like (кодек как вход + HuBERT k-means)
- E13: DinoSR-like (online EMA-кластеризация)

**Разложение эффектов (три пары сравнений):**
1. Эффект кодек-таргета: E2 vs E2c
2. Эффект архитектуры: E2c vs E3
3. Эффект multi-level: E3 vs E4/E5

**Probing-анализ:** линейный probe по каждому RVQ-уровню на фонемы, спикера, энергию. Эмпирическая опора для H3.

**Метрики:** WER/PER (10h/100h CTC), total GPU-hours до фикс. качества, masked-acc и perplexity по каждой кодбуке, throughput, пиковая память, ГБ на диске, 5 сидов, mean±std, paired bootstrap.

## 5. Обзор патча `codec_targets.patch`

### 5.1 Что сделано

- `src/codec/targets.py` — `CodecTargetExtractor`, `DACCodec`, `StubCodec`, `build_codec`.
- `dump_codec_labels.py` — offline-дамп в `.km`-формат.
- `test_codec_targets.py` — 5 unit-тестов.
- `train_hubert.py` — флаги `--targets`, `--codec`, `--codec-num-q`, `--codebook-weights`, `--q-curriculum-steps`, `--precision`; фикс DDP для одного GPU.
- `hubert_lightning_module.py` — online-таргеты, per-codebook метрики, throughput, память, curriculum.
- `hubert_model.py` — `codebook_weights`, `active_heads`, проекция только masked-кадров.
- `prediction_head.py` — cosine → normalize + matmul (экономия ~4.6 ГБ на голову).
- `masking.py` — векторизованное span-маскирование.
- `config.py`, `hubert_data_module.py`, `hubert_transforms.py` — поддержка online-режима.

### 5.2 Критические баги

1. **Рассогласование длин `targets` и `encoded`.** В `HubertModel.forward` делается `targets[masked]`, где `masked` имеет форму выхода CNN HuBERT, а `targets` — форму выхода кодека. Для DAC-16k оба 50 Гц, но округление при ресемплинге или разные padding-политики могут дать ±1 кадр. Нужен assert или slice.
2. **Сигнатура `waveform_16k`.** В `WaveOnlyPretrainTransform` вызов с двумя аргументами, в `dump_codec_labels.py` — с одним. Один из них неверен.
3. **`CodecTargetExtractor.forward` игнорирует `lengths`.** Коды на padding-позициях — мусор. Нужно обрезать `codes` по valid-длинам.

### 5.3 Другие проблемы

- Нет сброса `_t_last` на границе эпохи — throughput врёт на первом батче.
- `int(round(...))` при ресемплинге даёт floor, а не nearest.
- `dump_codec_labels.py` без батчинга, прогресса и обработки ошибок.
- EnCodec/Mimi не реализованы — только точка расширения.
- `bf16-mixed` для кодека не обрабатывается — нужен каст в fp32.
- `active_heads` — обычный атрибут, не буфер (ок для DDP, но хрупко).
- `codebook_weights` не в state_dict — при загрузке чекпоинта берутся из конфига.

### 5.4 Чего не хватает в тестах

- Реальный DAC.
- Ресемплинг индексов для не-50 Гц.
- Полный цикл `HubertPretrainModule` с `codec-online`.
- DDP с кодеком вне state_dict.
- Эквивалентность offline/online.
- bf16 для кодека.
- `dump_codec_labels.py`.

## 6. Следующие шаги

**Must fix (до любых экспериментов):**
1. Выравнивание `targets` и `encoded` по времени (assert + slice).
2. Сигнатура `waveform_16k` в `WaveOnlyPretrainTransform`.
3. Обрезка `codes` по valid-длинам.

**Should fix:**
4. bf16 → fp32 для входа кодека.
5. Сброс `_t_last` по эпохам.
6. tqdm/error handling в `dump_codec_labels.py`.

**Nice to have:**
7. Nearest вместо floor при ресемплинге.
8. Больше тестов (ресемплинг, offline/online эквивалентность).

**Первый приоритет:** Docker-окружение с Lightning/torchaudio/dac, `test_hubert.py` зелёный, smoke-test на 1 GPU с 1h LibriSpeech, проверка offline/online эквивалентности на реальном DAC.

## 7. Ожидаемые результаты (гипотезы)

- H1 (ослаблена): WER/PER в пределах ±10% отн. от HuBERT-iter2 при 1 итерации и 0 ГБ фич.
- Выигрыш по GPU-часам ~1.5–2× относительно полного пайплайна E0+E1.
- E2c покажет, что кодек-таргеты сами по себе не лучше random quantizer'а на BEST-RQ-архитектуре (или лучше — тогда это интересно).
- E3 vs E4/E5 покажет вклад multi-level RVQ.
- Основная деградация, если будет — на поздних слоях/ASR, поэтому E9 — страховка.

## 8. Риски

1. Кодек-таргеты шумные по фонетике → WER хуже k-means (результат EnCodecMAE). Митигация: веса, curriculum, E9, probing-анализ.
2. Пересечение с BEST-RQ/Codec2Vec/DinoSR. Митигация: E2c/E2m/E12/E13 + разложение эффектов.
3. «Это просто BEST-RQ с другим квантизатором». Митигация: честно указать идентичность таргетов, позиционировать как efficiency study.
4. Ресемплинг дискретных индексов даёт артефакты. Митигация: DAC-16k (50 Гц) как основной.
5. 3 сидов мало → 5 сидов.
6. H5 на abstract → полное чтение 2607.26350.

## 9. Итог

Патч — рабочий каркас, а не готовая реализация. Инженерно грамотный: изоляция кодека, фиксы реальных багов, аккуратные оптимизации, есть тесты на критические пути. Но до запуска экспериментов нужно закрыть критичные баги и провести smoke-test на реальном DAC. Если H1 не подтвердится, останется корректный efficiency study с честными baseline.
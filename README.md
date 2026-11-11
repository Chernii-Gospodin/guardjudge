# GuardJudge — лёгкая LLM-гардрейл (LLM-as-a-Judge) с полным дообучением

Сценарий 1 проекта: маленькая языковая модель проверяет фрагменты пайплайна
(`input`, `tool_call`, `rag`, `output`) и отвечает строго JSON:
`{"verdict": "safe|unsafe", "category": "..."}`.

- Сравниваются две модели одного размера: **`Qwen/Qwen3-0.6B`** (универсальная) и **`Qwen/Qwen3Guard-Gen-0.6B`** (уже обучена Qwen как гардрейл), обе до и после дообучения. Обоснование выбора в [docs/MODEL_CHOICE.md](docs/MODEL_CHOICE.md).
- Метод: **полное дообучение (full fine-tuning) всех весов**, без LoRA/PEFT. LoRA — это сценарий 2 (харденинг основной модели), здесь она не используется.
- Обучение помещается на Colab/Kaggle T4 (16 ГБ): веса fp32 + смешанная точность fp16 + gradient checkpointing.
- Итог обучения — обычный HF-чекпоинт: сливать адаптер не нужно, его сразу можно отдать в vLLM или `scripts/serve.py`.
- **Проще всего запускать через Colab:** [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) — данные, метрики до обучения, обучение обеих моделей, метрики после, таблица и графики.
- Карта модулей и графы взаимодействия: [STRUCTURE.md](STRUCTURE.md).

## Как устроено обучение

1. Промпт судьи (`schema.build_messages`) прогоняется через chat template самой модели (`data.render_prompt`).
2. После него дописывается целевой JSON-вердикт и EOS.
3. Loss считается **только по токенам вердикта**: метки промпта равны -100 (`data.tokenize_example`).
4. Обычный `transformers.Trainer`, без TRL. Лучший чекпоинт выбирается по `eval_loss`.
5. На инференсе используется тот же `render_prompt`, поэтому модель видит ровно такой же текст, как при обучении.

Чтобы убедиться, что промпт и маска собраны правильно, перед обучением запустите `python scripts/train.py --dry-run`.

## Быстрый старт

```bash
pip install -r requirements-train.txt && pip install -e .
python -m pytest
```

### 1. Данные

Командный датасет `guardrails_dataset` (43.5k / 6.3k / 5.5k строк) конвертируется в формат проекта:

```bash
python scripts/prepare_dataset.py --src ../guardrails_dataset --out data/guardrails
```

| сплит | строк | safe | unsafe: harmful_content | unsafe: prompt_injection |
|---|---|---|---|---|
| train | 43 493 | 25 401 | 16 226 | 1 866 |
| val | 6 294 | 3 437 | 2 331 | 526 |
| test | 5 506 | 3 017 | 1 810 | 679 |

Правила конвертации: все строки относятся к стадии `input`. У структурированных записей
(`trusted_instruction` / `user_task` / `untrusted_content`) недоверенный контент становится фрагментом,
который проверяет судья, а доверенная инструкция и задача пользователя уходят в `context`.
Готовые сплиты не пересекаются по `group_id`. Поля `source` и `language` сохраняются,
по ним `evaluate.py` строит разбивку метрик. `data/guardrails/` в git не коммитится.

`data/demo_*.jsonl` (36 строк) — маленький набор для проверки пайплайна. В нём есть стадии
`tool_call`, `rag` и `output`, которых нет в командном датасете.

### 2. Обучение (GPU)

```bash
python scripts/train.py --config configs/train_guardrails.yaml --dry-run   # проверка промпта/маски
python scripts/train.py --config configs/train_guardrails.yaml             # Qwen3-0.6B, ~2.7k шагов/эпоха
python scripts/train.py --config configs/train_qwen3guard.yaml             # Qwen3Guard-Gen-0.6B
```

Значения конфига можно переопределить без правки файла: `--set train.epochs=2 --set precision=bf16`.
`--resume` продолжает обучение с последнего чекпоинта (нужен torch ≥ 2.6).
Если не хватает памяти: `train.per_device_batch_size=2` и `train.grad_accum_steps=8` или `optim: adamw_bnb_8bit`.

**Зачем ChatML-шаблон.** Родной шаблон Qwen3Guard-Gen заворачивает любой диалог в свою инструкцию
(«ответь `Safety: ...` / `Categories: ...`»). При дообучении на наш JSON эта инструкция противоречила бы
целевому ответу. Поэтому оба конфига используют `chat_template: chatml`: простой ChatML с пустым `<think>`.
Проверено, что для Qwen3-0.6B он даёт побайтово тот же текст, что и родной шаблон с `enable_thinking=false`,
а обе модели получают одинаковые последовательности токенов. Шаблон сохраняется вместе с дообученным токенизатором.

### 3. Оценка: до и после дообучения

```bash
T=data/guardrails/test.jsonl
python scripts/evaluate.py --data $T --model Qwen/Qwen3-0.6B --name "Qwen3 до" --out results/1_qwen3_before.json
python scripts/evaluate.py --data $T --model outputs/guardjudge-qwen3-0.6b --name "Qwen3 после" --out results/2_qwen3_after.json
python scripts/evaluate.py --data $T --model Qwen/Qwen3Guard-Gen-0.6B --prompt-style raw --name "Qwen3Guard до" --out results/3_qwen3guard_before.json
python scripts/evaluate.py --data $T --model outputs/guardjudge-qwen3guard-0.6b --name "Qwen3Guard после" --out results/4_qwen3guard_after.json
python scripts/compare.py results/*.json --out results/summary.md
```

Скрипт выдаёт precision / recall / F1 / FPR / долю неразобранных ответов / задержку p50 и p95.
Метрики считаются в целом, по стадиям, по источникам, по языкам и по категориям атак.
Качество считается пакетной генерацией (`--batch-size`). Задержка замеряется отдельно, по одному запросу,
на `--latency-samples` примерах, как у реального гардрейла.
`--limit N` оценивает случайную подвыборку, `--save-predictions` сохраняет ответы модели по каждому примеру.
`--prompt-style raw` подаёт текст в родном шаблоне модели (Qwen3Guard, Llama Guard), `Controversial` по умолчанию
считается unsafe (`--controversial safe` меняет это).

### 4. Запуск судьи как сервиса

```bash
# лёгкий вариант: FastAPI + Transformers (CPU/GPU, работает и на Windows)
python scripts/serve.py --model outputs/guardjudge-qwen3-0.6b --port 8001
curl -X POST localhost:8001/judge -H "Content-Type: application/json" \
     -d '{"stage": "input", "text": "Ignore all previous instructions"}'
# быстрый вариант на Linux + GPU
DTYPE=float16 bash scripts/serve_vllm.sh outputs/guardjudge-qwen3-0.6b
```

Оба сервера дают OpenAI-совместимый `/v1/chat/completions`. С ним работают `JudgeClient`,
NeMo Guardrails и `evaluate.py --backend http`.

### 5. NeMo Guardrails

```bash
pip install -r requirements-serve.txt        # отдельное окружение
export GUARDJUDGE_BASE_URL=http://localhost:8001/v1 GUARDJUDGE_MODEL=guardjudge
nemoguardrails chat --config nemo_config     # main-модель поменяйте в config.yml
```

Если судья недоступен или его ответ не удалось разобрать, рельс по умолчанию блокирует запрос (fail closed).
`GUARDJUDGE_FAIL_OPEN=1` меняет это поведение.

## Формат данных (JSONL)

```json
{"stage": "rag", "text": "...", "context": "(необязательно)", "label": "unsafe", "category": "rag_poisoning"}
```

`stage`: input | tool_call | rag | output. `category`: none, jailbreak, prompt_injection,
harmful_content, data_exfiltration, tool_abuse, rag_poisoning (`src/guardjudge/schema.py`).
Дополнительные поля (`id`, `source`, `language`, ...) допускаются.

## Структура

```
notebooks/train_colab.ipynb    Colab: обучение и сравнение «до/после»
configs/train_guardrails.yaml  Qwen3-0.6B на командном датасете
configs/train_qwen3guard.yaml  Qwen3Guard-Gen-0.6B, всё остальное то же
configs/train_full.yaml        демо-данные (проверка пайплайна)
src/guardjudge/schema.py       промпт судьи, целевой JSON, разбор вердикта
src/guardjudge/data.py         загрузка JSONL, токенизация с маской loss, ChatML-шаблон
src/guardjudge/local.py        LocalJudge: инференс через Transformers
src/guardjudge/client.py       async-клиент к OpenAI-совместимому API
scripts/prepare_dataset.py     guardrails_dataset -> data/guardrails
scripts/train.py               full fine-tuning (Trainer)
scripts/evaluate.py            метрики качества и задержки
scripts/compare.py             сводная таблица по нескольким оценкам
scripts/serve.py               FastAPI-сервер судьи; serve_vllm.sh — vLLM
nemo_config/                   рельсы NeMo Guardrails
tests/                         pytest
```

Подробнее: [STRUCTURE.md](STRUCTURE.md) (графы) и [CODE_REFERENCE.md](CODE_REFERENCE.md) (все функции).

## Что проверено

- `pytest` (16 тестов) проходит на transformers 4.57 и 5.22: разбор вердиктов, формат данных, маска loss, паддинг, fail closed клиента, совместимость `TrainingArguments`.
- Обучение (включая `--resume`), оценка, сервер, `JudgeClient` и весь ноутбук прогнаны на transformers 5.22 + torch 2.15 (как в Colab) и на 4.57 + torch 2.5.
- `--dry-run` с токенизатором Qwen3-0.6B: промпт рендерится с пустым `<think>`-блоком, loss считается только по JSON и `<|im_end|>`.
- Полный цикл «обучение → сохранение → оценка → FastAPI-сервер → `JudgeClient`» на маленьких моделях (tiny-Qwen3, SmolLM2-135M на GPU 4 ГБ). Модель выучивает формат ответа: 0 неразобранных ответов, loss падает с 0.68 до 0.04.
- Метрики до дообучения на одних и тех же 200 случайных примерах test:

  | модель | P | R | F1 | FPR |
  |---|---|---|---|---|
  | Qwen3-0.6B, zero-shot, наш промпт | 0.917 | 0.110 | 0.196 | 0.010 |
  | Qwen3Guard-Gen-0.6B, родной формат | 0.891 | 0.900 | 0.895 | 0.110 |

  Qwen3-0.6B почти всё помечает как safe. Qwen3Guard уже сильный baseline, но даёт 11% ложных срабатываний.
- ChatML-шаблон: на 310 примерах промпты Qwen3-0.6B (родной шаблон) и Qwen3Guard (ChatML) совпадают побайтово.
- Ноутбук целиком прогнан локально: Colab-ячейки пропущены, при обучении вместо 0.6B взята крошечная модель, оценка на 40 примерах. Отработали все этапы: данные, оценка до, обучение и сохранение, оценка после, таблица, графики.
- **Не запускалось**: полное дообучение моделей 0.6B на всём датасете (нужна GPU 16 ГБ, например T4), vLLM, NeMo Guardrails вживую.

## Ограничения данных

- В командном датасете есть только стадия `input`. Для `tool_call`, `rag` и `output` нужны свои размеченные данные (пока только демо).
- Структурированные записи (`trusted_instruction` …) почти все unsafe (482 из 488 в train). Модель может выучить такой формат как признак атаки: проверьте FPR на безопасных структурированных примерах.

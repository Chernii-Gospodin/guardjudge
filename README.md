# GuardJudge — дообучаемая лёгкая LLM-гардрейл (LLM-as-a-Judge) для NeMo Guardrails

Проект для сценария 1: лёгкая модель-судья проверяет фрагменты пайплайна
(`input`, `tool_call`, `rag`, `output`) и отвечает строго JSON:
`{"verdict": "safe|unsafe", "category": "..."}`.

Базовая модель по умолчанию: `Qwen/Qwen3Guard-Gen-4B` (есть вариант 0.6B).
Обоснование выбора — в Google Doc.

## Статус проверки

- Проверено: unit-тесты (`pytest`, разбор вердиктов, формат данных), синтаксис всех скриптов.
- **Не запускалось** (нужны GPU и доступ к Hugging Face): обучение, слияние адаптера, vLLM, интеграция с NeMo Guardrails, `evaluate.py` на живой модели.
- Нужно проверить на вашем железе: chat template Qwen3Guard-Gen (специальный шаблон классификации) совместим с нашим промптом. Если обучение даёт мусор, возьмите обычную instruct-модель (`model_name` в конфиге) или подгоните шаблон. Быстрая проверка: `python scripts/smoke_test.py`.
- Retrieval-рельс (`guardjudge_filter_chunks`) и синтаксис Colang проверьте на вашей версии NeMo Guardrails командой `nemoguardrails chat --config nemo_config`. Рельс для tool calls (execution rails) пока не сделан (TODO).
- Демо-данные (36 строк) нужны только для проверки пайплайна, метрики на них ничего не значат.

## Быстрый старт

```bash
pip install -r requirements-train.txt && pip install -e .
python -m pytest                                   # проверка парсеров и данных

# 0. Smoke-тест: шаблон чата и разбор вердикта до долгого обучения
python scripts/smoke_test.py --model Qwen/Qwen3Guard-Gen-0.6B

# 1. Обучение LoRA (plain LoRA + fp16, совместимо с T4; GPU)
python scripts/train.py --config configs/train_lora.yaml

# 2. Слияние адаптера и запуск судьи
python scripts/merge_lora.py --base Qwen/Qwen3Guard-Gen-4B --dtype fp16 \
    --adapter outputs/guardjudge-qwen3guard-4b-lora --out merged/guardjudge-4b
DTYPE=float16 bash scripts/serve_vllm.sh merged/guardjudge-4b   # OpenAI-совместимый API на :8001

# 3. Оценка: дообученная модель и baseline без дообучения
python scripts/evaluate.py --data data/demo_test.jsonl --model guardjudge --out results/tuned.json
vllm serve Qwen/Qwen3Guard-Gen-4B --port 8002 &    # baseline
python scripts/evaluate.py --data data/demo_test.jsonl --base-url http://localhost:8002/v1 \
    --model Qwen/Qwen3Guard-Gen-4B --prompt-style raw --out results/baseline.json

# 4. NeMo Guardrails с судьёй (в отдельном окружении от обучения)
pip install -r requirements-serve.txt
export GUARDJUDGE_BASE_URL=http://localhost:8001/v1 GUARDJUDGE_MODEL=guardjudge
nemoguardrails chat --config nemo_config           # поменяйте main-модель в config.yml
```

## Формат данных (JSONL)

```json
{"stage": "rag", "text": "...", "context": "(необязательно)", "label": "unsafe", "category": "rag_poisoning"}
```

`stage`: input | tool_call | rag | output. `category`: none, jailbreak, prompt_injection,
harmful_content, data_exfiltration, tool_abuse, rag_poisoning (`src/guardjudge/schema.py`).
Для настоящего обучения замените `data/demo_*.jsonl` своими данными; отложите в тест целые семейства
атак и безопасные запросы того же стиля (для контроля ложных срабатываний).

## Структура

```
configs/train_lora.yaml    параметры LoRA (plain LoRA + fp16, r=16, alpha=32, lr=1e-4)
src/guardjudge/            промпт, парсер вердикта, загрузка данных, async-клиент судьи
scripts/                   train.py, merge_lora.py, serve_vllm.sh, evaluate.py, smoke_test.py
nemo_config/               config.yml, rails.co, actions.py — рельсы NeMo Guardrails
data/                      демо-датасеты
tests/                     pytest
```

Если судья недоступен, рельс по умолчанию блокирует запрос (fail closed).
`GUARDJUDGE_FAIL_OPEN=1` меняет это поведение.

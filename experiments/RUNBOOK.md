# Единый запуск экспериментов

Все эксперименты запускаются одной командой — `scripts/launch_experiment.py`.
Она выбирает конфиг, автоматически подставляет корень датасета, создаёт папку
результатов и запускает обучение. Для матрицы seeds/losses/selectors тот же
launcher сам создаёт и исполняет задания; вручную вызывать `run.sh` не нужно.

## Один раз перед запуском

Активируй окружение Blackwell из [README экспериментов](README.md), проверь
`nvidia-smi`, затем укажи корень своих локальных весов:

```bash
source .venv-blackwell/bin/activate
export MODEL_DIR=/mnt/pretrained-models  # замени на фактический путь
```

Ожидаемая структура для скачанных моделей:

```text
/mnt/pretrained-models/
  FacebookAI/roberta-base/
  sentence-transformers/all-MiniLM-L6-v2/
  torchvision/checkpoints/<TorchVision checkpoint files>
```

Hugging Face папки должны содержать полный snapshot ревизии, указанной в
конфиге. Также поддерживается стандартный HF cache с папками
`models--<organization>--<repo>/snapshots/<revision>`. TorchVision веса должны
лежать в `torchvision/checkpoints/`. Когда передан `--models-dir`, launcher
использует только локальные файлы и сообщает об ошибке, если модель не найдена.

## Главное правило

```bash
python scripts/launch_experiment.py \
  --config <имя-конфига> \
  --models-dir "$MODEL_DIR"
```

Например, один обычный Omniglot retrieval-запуск:

```bash
python scripts/launch_experiment.py \
  --config omniglot_metric_dev \
  --models-dir "$MODEL_DIR"
```

Launcher сам использует `/home/arseniy/data/omniglot` и создаёт уникальную
папку в `experiments/runs/omniglot_metric_dev/`. Это полный бюджет конфига;
модель обучится и посчитает validation. Для вывода команды без запуска добавь
`--dry-run`.

## Пилот Omniglot

Один seed, один epoch и два train-step. Это настоящий запуск, но не итоговый
результат; validation всё равно считается на настроенном dev-разбиении.

```bash
python scripts/launch_experiment.py \
  --config omniglot_metric_dev \
  --models-dir "$MODEL_DIR" \
  --seed 1 --losses catml --selectors random \
  --examples-per-class 10 --epochs 1 --steps-per-epoch 2
```

## Матрица Omniglot

Одна команда строит матрицу CATML по пяти seeds и сразу запускает её:

```bash
python scripts/launch_experiment.py \
  --config omniglot_metric_dev \
  --models-dir "$MODEL_DIR" \
  --seeds 1,2,3,4,5 \
  --losses catml \
  --selectors random,apm,auto_clusters,apm_auto
```

Для triplet-матрицы используй тот же шаблон с параметрами:

```bash
--losses triplet --selectors random,semi_hard,batch_hard
```

После выбора метода по validation финальная class-disjoint оценка выглядит так:

```bash
python scripts/launch_experiment.py \
  --config omniglot_metric_dev \
  --models-dir "$MODEL_DIR" \
  --stage final --run-test \
  --seeds 1,2,3,4,5 \
  --losses catml --selectors random
```

Она переобучит выбранный метод на разрешённых train + validation алфавитах и
оценит его на отложенных Omniglot test-алфавитах. Для пилота или перебора
настроек `--run-test` не добавляй.

## Остальные Omniglot-постановки

| Что запускать | Команда/настройки |
|---|---|
| Closed-set classification | `--config omniglot_classification --models-dir "$MODEL_DIR"`; для итоговой оценки добавь `--run-test` |
| Ограничить обучение 5/10 объектами на класс | К classification-команде добавь `--examples-per-class 5,10` и, например, `--seeds 1,2,3,4,5` |
| 5-way 1-shot | `--config omniglot_fewshot_dev --models-dir "$MODEL_DIR" --run-test` после фиксации настроек |
| 5-way 5-shot | Та же команда с `--k-shot 5` |
| Авторская run-folder one-shot | `--config omniglot_run_folder_oneshot --models-dir "$MODEL_DIR" --run-test` |

Если нужен только validation, убери `--run-test`. Для 5-shot и любого списка
seeds/losses/selectors launcher создаст матрицу и исполнит её в том же вызове.
Для O5 проверь на вычислительном узле наличие
`/home/arseniy/data/omniglot/python/all_runs`; её соответствие официальному
набору заданий пока не подтверждено.

## Остальные датасеты и конфиги

Можно передать имя любого конфига из `experiments/configs/` — `.json` в имени
необязателен. Датасет launcher определяет сам:

| Датасет в конфиге | Корень по умолчанию |
|---|---|
| `clinc150` | `/home/arseniy/data/clinc150` |
| `cub200` | `/home/arseniy/data/CUB-200-2011` |
| `omniglot` | `/home/arseniy/data/omniglot` |
| `phd08` | `/home/arseniy/data/PHD08_dataset` |
| `kor_synthetic` | `/home/arseniy/data/kor_synthetic` |

P5 (`phd08_synthetic_2350` и `phd08_synthetic_full`) автоматически получает
KorSynthetic как train и PHD08 как evaluation data. Итоговую оценку включай
флагом `--run-test`. Если данные на другой машине лежат в других местах,
переопредели их через `--data-root` и для P5 через `--eval-data-root`.

Полезные общие параметры:

- `--seed 7` задаёт один seed; `--seeds 1,2,3,4,5` создаёт серию запусков.
- `--losses catml,triplet` и `--selectors random,apm_auto` задают сравниваемые методы.
- `--examples-per-class all,5,10` задаёт обучающие бюджеты.
- `--k-shot 5` меняет K в few-shot validation и test.
- `--epochs 1 --steps-per-epoch 2` сокращают пилот.
- `--output` и `--output-dir` задают свои места для одного запуска и матрицы.
- `--resume <run>/last.pt --output <тот-же-run>` продолжает одиночный запуск.

Состояние, метрики и resolved config сохраняются в `experiments/runs/` для
одиночного запуска и `experiments/jobs/<matrix>/runs/` для матрицы. Команда с
матрицей пишет все seed/loss/selector результаты последовательно и завершится
при первой ошибке.

## Границы реализованной части планов

Текущие конфиги запускают основные классификационные, retrieval, few-shot,
CLINC open-set и P5 сценарии, но не все расширенные ablation-ы и baselines из
`EXPERIMENT_PLAN_*.md`. В частности, PHD08 source/font-disjoint оценка пока
невозможна без идентификаторов источника; точный официальный протокол Omniglot
O5 также не подтверждён. Не включай test-результаты в подбор гиперпараметров.

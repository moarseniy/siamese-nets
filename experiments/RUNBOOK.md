# Запуск PhD-экспериментов

В репозитории одна схема конфигов в `experiments/configs/` и один trainer —
корневой `train.py`. Одиночные запуски идут напрямую через trainer; заранее
определённые серии и матрицы — через `scripts/run_experiment_batch.py`.

## Окружение и модели

```bash
source .venv/bin/activate
export MODEL_DIR=/путь/к/локальным-моделям
nvidia-smi
```

Зависимости закреплены в корневом `requirements.txt`. При передаче `--models-dir`
все веса должны уже находиться в локальных папках TorchVision или Hugging Face;
trainer не скачивает веса незаметно.

## Одиночный запуск

Каждому запуску передавай отдельную выходную папку. Конфиг, checkpoint, логи и
метрики сохраняются вместе:

```bash
python train.py \
  --config experiments/configs/phd08_metric_dev.json \
  --data-root /путь/к/PHD08_dataset \
  --models-dir "$MODEL_DIR" \
  --output experiments/runs/phd08_metric_dev/seed1
```

`--run-test` запускает test-протокол из конфига после выбора checkpoint по
validation. `--resume <output>/last.pt` продолжает обучение. Конфиг не
преобразуется и не получает значения из старых схем: отсутствующие обязательные
поля и неизвестные ключи приводят к ошибке.

Предобученную или уже обученную модель можно оценить без обучения:

```bash
python train.py \
  --config experiments/configs/clinc150_sentence_transformer_frozen.json \
  --data-root /путь/к/clinc150 \
  --models-dir "$MODEL_DIR" \
  --output experiments/runs/clinc150_frozen/validation.json \
  --evaluate-pretrained --split validation
```

Для open-set оценки `test` добавь `--calibrate-with-validation`, чтобы порог
был выбран только на validation.

## Матрицы

Сначала можно сформировать и проверить план без выполнения:

```bash
python scripts/run_experiment_batch.py \
  --series p1 --phase development \
  --data-root /путь/к/PHD08_dataset \
  --models-dir "$MODEL_DIR" \
  --seeds 1,2,3,4,5 --dry-run
```

После проверки запусти очередь, убрав `--dry-run`. Batch runner хранит статус
каждой работы и позволяет запросить `--status` или повторить ошибки через
`--retry-failed`.

| Серии | Датасет | Назначение |
|---|---|---|
| `p1`, `p2`, `p3`, `p4`, `p5` | PHD08 и KorSynthetic | Закрытая классификация, retrieval, few-shot и перенос с синтетики |
| `o1`–`o5` | Omniglot | Классификация, retrieval, few-shot и run-folder one-shot |
| `c1`–`c4` | CUB-200-2011 | Классификация, retrieval и few-shot |
| `l1`–`l6` | CLINC150 | Классификация, retrieval, few-shot и open-set |

PHD08 серии запускаются по отдельности. P5 требует корень обоих наборов:

```bash
python scripts/run_experiment_batch.py \
  --series p5 --phase development \
  --data-root /путь/к/PHD08_dataset \
  --kor-root /путь/к/kor_synthetic \
  --models-dir "$MODEL_DIR" \
  --seeds 1,2,3,4,5 --output-dir experiments/jobs/phd08_p5
```

`--phase final` используй только после фиксации протокола и настроек по
development/validation. Для P4 передай завершённую очередь P3 через
`--source-batch`. Few-shot серии `o4`, `c4`, `l4` требуют завершённую source
очередь той же фазы. O5 принимает `--oneshot-root`, если папка `all_runs` не
находится в стандартном корне Omniglot.

## Матрица PHD08

P1/P2 сравнивают CATML с Random, APM, Auto-clustering и APM+Auto-clustering,
Triplet с Random, Semi-hard и Batch-hard, а также Cross-entropy. P3 фиксирует
class-disjoint retrieval; P4 оценивает сохранённые P3-модели на 1-shot/5-shot;
P5 оценивает модели, обученные на KorSynthetic, на PHD08. Точные разбиения,
бюджеты и ограничения приведены в [плане PHD08](../EXPERIMENT_PLAN_PHD08.md).

## Манифесты

Манифесты закрепляют состав классов, протокол и контрольные суммы данных. После
изменения исходных данных пересоздай соответствующий manifest через
`scripts/prepare_manifests.py`, затем проверь разбиение до обучения. Runner
отклоняет несовпадение состава классов и пересечение классов train/validation.

Протоколы и ограничения датасетов описаны в
[планах PHD08](../EXPERIMENT_PLAN_PHD08.md),
[Omniglot](../EXPERIMENT_PLAN_OMNIGLOT.md), [CUB](../EXPERIMENT_PLAN_CUB.md) и
[CLINC150](../EXPERIMENT_PLAN_CLINC150.md).

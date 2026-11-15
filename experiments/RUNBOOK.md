# Единый запуск экспериментов

Все эксперименты запускаются одной командой — `scripts/launch_experiment.py`.
Она выбирает конфиг, автоматически подставляет корень данных, создаёт папку
результатов и начинает обучение. Для матрицы начальных значений генератора,
функций потерь и способов выборки тот же сценарий сам создаёт и запускает
задания; вручную вызывать `run.sh` не нужно.

В тексте ниже «обучение», «проверка» и «итоговая оценка» соответствуют разделам
`validation` и `test` в конфигах. Имена параметров и режимов в командах оставлены
в исходном виде, который понимает программа.

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
  torchvision/checkpoints/<файлы весов TorchVision>
```

Папки Hugging Face должны содержать полный снимок нужной ревизии, указанной в
конфиге. Также поддерживается стандартный кэш HF с папками
`models--<организация>--<репозиторий>/snapshots/<ревизия>`. Веса TorchVision должны
лежать в `torchvision/checkpoints/`. Когда передан `--models-dir`, сценарий запуска
использует только локальные файлы и сообщает об ошибке, если модель не найдена.

## Главное правило

```bash
python scripts/launch_experiment.py \
  --config <имя-конфига> \
  --models-dir "$MODEL_DIR"
```

Например, один запуск метрического поиска похожих изображений Omniglot:

```bash
python scripts/launch_experiment.py \
  --config omniglot_metric_dev \
  --models-dir "$MODEL_DIR"
```

Сценарий сам использует `/home/arseniy/data/omniglot` и создаёт уникальную
папку в `experiments/runs/omniglot_metric_dev/`. Применяется полный бюджет
конфига: модель обучится и будет проверена на отложенном проверочном наборе.
Чтобы только показать команду, добавь `--dry-run`.

## Пилот Omniglot

Один seed (начальное значение генератора случайных чисел), одна эпоха и два шага
обучения. Это настоящий запуск, но не итоговый
результат; модель всё равно будет проверена на отложенном разбиении для разработки.

```bash
python scripts/launch_experiment.py \
  --config omniglot_metric_dev \
  --models-dir "$MODEL_DIR" \
  --seed 1 --losses catml --selectors random \
  --examples-per-class 10 --epochs 1 --steps-per-epoch 2
```

## Матрица Omniglot

Одна команда строит и запускает матрицу CATML для пяти начальных значений
генератора случайных чисел:

```bash
python scripts/launch_experiment.py \
  --config omniglot_metric_dev \
  --models-dir "$MODEL_DIR" \
  --seeds 1,2,3,4,5 \
  --losses catml \
  --selectors random,apm,auto_clusters,apm_auto
```

Для матрицы с триплетной функцией потерь используй тот же шаблон с параметрами:

```bash
--losses triplet --selectors random,semi_hard,batch_hard
```

После выбора метода по проверочному набору итоговая оценка на классах, которые
не встречались при обучении, запускается так:

```bash
python scripts/launch_experiment.py \
  --config omniglot_metric_dev \
  --models-dir "$MODEL_DIR" \
  --stage final --run-test \
  --seeds 1,2,3,4,5 \
  --losses catml --selectors random
```

Модель будет обучена на разрешённых обучающих и проверочных алфавитах, затем
оценена на отложенных тестовых алфавитах Omniglot. Для пилота и перебора
настроек `--run-test` не добавляй.

## Остальные Omniglot-постановки

| Эксперимент | Команда или настройка |
|---|---|
| Классификация на известных классах | `--config omniglot_classification --models-dir "$MODEL_DIR"`; для итоговой оценки добавь `--run-test` |
| Обучение на 5 или 10 изображениях класса | К команде классификации добавь `--examples-per-class 5,10` и, например, `--seeds 1,2,3,4,5` |
| Эпизодическая классификация: 5 классов, 1 опорное изображение каждого | `--config omniglot_fewshot_dev --models-dir "$MODEL_DIR" --run-test` после фиксации настроек |
| Эпизодическая классификация: 5 классов, 5 опорных изображений каждого | Та же команда с `--k-shot 5` |
| Оценка исходной задачи с одним опорным примером из папок заданий | `--config omniglot_run_folder_oneshot --models-dir "$MODEL_DIR" --run-test` |

Если нужна только проверка без итоговой оценки, убери `--run-test`. Для режима
с пятью опорными изображениями и любого списка начальных значений, функций потерь
и способов выборки сценарий создаст матрицу и запустит её в том же вызове. Для O5
проверь на вычислительном узле наличие
`/home/arseniy/data/omniglot/python/all_runs`; её соответствие официальному
набору заданий пока не подтверждено.

## Остальные датасеты и конфиги

Можно передать имя любого конфига из `experiments/configs/`; окончание `.json`
указывать необязательно. Сценарий сам определит датасет:

| Датасет в конфиге | Корень по умолчанию |
|---|---|
| `clinc150` | `/home/arseniy/data/clinc150` |
| `cub200` | `/home/arseniy/data/CUB-200-2011` |
| `omniglot` | `/home/arseniy/data/omniglot` |
| `phd08` | `/home/arseniy/data/PHD08_dataset` |
| `kor_synthetic` | `/home/arseniy/data/kor_synthetic` |

P5 автоматически использует KorSynthetic для обучения и PHD08 для итоговой
оценки. Если данные на вычислительном узле находятся в других местах, задай их
через `--data-root`, а для P5 также через `--eval-data-root`.

Полезные общие параметры:

- `--seed 7` задаёт одно начальное значение генератора случайных чисел;
  `--seeds 1,2,3,4,5` задаёт серию запусков.
- `--losses catml,triplet` задаёт функции потерь, а
  `--selectors random,apm_auto` — способы выборки.
- `--examples-per-class all,5,10` задаёт число доступных обучающих примеров на класс.
- `--k-shot 5` задаёт пять опорных примеров класса для проверки и итоговой оценки
  в режиме классификации по нескольким примерам.
- `--epochs 1 --steps-per-epoch 2` сокращают пилотный запуск.
- `--output` и `--output-dir` задают свои места для одного запуска и матрицы.
- `--resume <run>/last.pt --output <тот-же-run>` продолжает одиночный запуск.

Состояние модели, метрики и итоговый конфиг сохраняются в `experiments/runs/`
для одиночного запуска и `experiments/jobs/<матрица>/runs/` для серии запусков.
Задания серии выполняются последовательно; первая ошибка останавливает очередь.

## Границы реализованной части планов

Текущая основная PHD08-матрица соответствует восьми сравнениям из плана. Более
широкие baseline-строки и абляции из раздела 6 плана (PFML, Proxy-AN,
обучаемый ProtoNet и другие селекторы) пока не добавлены. PHD08 split исключает
пересечение восстановленных по именам файлов групп; font-disjoint проверка не
проводится. Точный официальный протокол Omniglot O5 также не подтверждён. Не
используй результаты итоговой оценки для подбора гиперпараметров.

## Матрицы PHD08

### P1/P2: известные классы, полный и ограниченный бюджет

Пять seeds × восемь строк × три бюджета дают 120 обучений. Четыре строки CATML
сравнивают Random, APM, Auto-clustering и их совместный режим; три строки
Triplet — Random, Semi-hard и Batch-hard; Cross-entropy служит отдельным
классификационным baseline. Для P2 бюджеты 5 и 10 выбирают фиксированные
представители разных filename-derived групп; 5 вложен в 10.

```bash
python scripts/launch_experiment.py \
  --config phd08_metric_closed_dev \
  --cross-entropy-config phd08_classification \
  --phd08-baseline \
  --models-dir "$MODEL_DIR" \
  --seeds 1,2,3,4,5 \
  --examples-per-class all,5,10 \
  --run-test \
  --output-dir experiments/jobs/phd08_p1p2
```

### P3: retrieval новых классов

Итоговая матрица обучается на 1 880 train+validation классах без выбора по test,
затем оценивает Recall@K на 470 отложенных классах. Запусти 40 обучений:

```bash
python scripts/launch_experiment.py \
  --config phd08_metric_dev \
  --cross-entropy-config phd08_metric_ce_dev \
  --phd08-baseline \
  --models-dir "$MODEL_DIR" \
  --seeds 1,2,3,4,5 \
  --stage final \
  --run-test \
  --output-dir experiments/jobs/phd08_p3
```

Cross-entropy здесь обучает классификационную голову на train-классах, но при
итоговой оценке голова не используется: сравниваются эмбеддинги, как у остальных
строк.

### P4: 1-shot и 5-shot на checkpoint P3

P4 не обучает отдельную сеть. Для каждого выбранного P3 checkpoint запускай две
оценки. Например, для одного CATML seed 1:

```bash
P3_RUN=experiments/jobs/phd08_p3/runs/phd08_final_catml_random_all_seed1
python scripts/launch_experiment.py \
  --config phd08_fewshot_final_k1 --models-dir "$MODEL_DIR" \
  --checkpoint "$P3_RUN/last.pt" --split test \
  --output experiments/jobs/phd08_p4/catml_random_seed1_k1.json
python scripts/launch_experiment.py \
  --config phd08_fewshot_final_k5 --models-dir "$MODEL_DIR" \
  --checkpoint "$P3_RUN/last.pt" --split test \
  --output experiments/jobs/phd08_p4/catml_random_seed1_k5.json
```

Замени путь `P3_RUN` на соответствующий метод и seed. Валидационные эпизоды
используются для настройки, test-эпизоды — только в итоговом отчёте. Сохраняются
точные IDs изображений каждого эпизода.

### P5: KorSynthetic → PHD08

Каждая из двух команд ниже создаёт по 40 обучений: две конфигурации алфавита ×
восемь строк × пять seeds. В PHD08-классификации рассматриваются все 11 172
обучающих класса в режиме `full`; галерея ближайшего эталона содержит один
детерминированный синтетический train-образец на класс. Для выбора checkpoint
используются 10 детерминированных validation-примеров на класс; основная
метрика выбора — точность ближайшего эталона для всех строк. Для внешнего
датасета используются стандартные корни из таблицы выше; при необходимости
добавь `--eval-data-root`.

```bash
python scripts/launch_experiment.py \
  --config phd08_synthetic_metric_2350 \
  --cross-entropy-config phd08_synthetic_2350 \
  --phd08-baseline --models-dir "$MODEL_DIR" \
  --seeds 1,2,3,4,5 --run-test \
  --output-dir experiments/jobs/phd08_p5_2350

python scripts/launch_experiment.py \
  --config phd08_synthetic_metric_full \
  --cross-entropy-config phd08_synthetic_full \
  --phd08-baseline --models-dir "$MODEL_DIR" \
  --seeds 1,2,3,4,5 --run-test \
  --output-dir experiments/jobs/phd08_p5_full
```

В строке Cross-entropy итоговый отчёт содержит logits/accuracy и
nearest-exemplar метрики; в метрических строках выводятся nearest-exemplar и
nearest-centroid. Для `full` множество допустимых ответов — все 11 172
обучающих символа; тестовые изображения относятся только к 2 350 классам PHD08.

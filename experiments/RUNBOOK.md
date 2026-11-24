# Единый запуск экспериментов

Параллельные очереди для основных серий запускаются через
`scripts/run_experiment_batch.py`; старый `scripts/launch_experiment.py`
остаётся доступен для одиночных запусков и последовательных матриц.
Обе команды выбирают конфиги и создают каталоги результатов.

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

## Очередь для всех датасетов

Общий launcher поддерживает PHD08 и HangulPrinted/KorSynthetic, Omniglot,
CUB-200-2011 и CLINC150. Указывай корень конкретного датасета в `--data-root`;
для P5 это PHD08, а HangulPrinted/KorSynthetic передаётся отдельно в `--kor-root`.
Настройки `--gpus`, `--workers-per-gpu`, `--seeds`, `--epochs`,
`--steps-per-epoch`, `--dry-run`, `--status` и `--retry-failed` работают так же,
как в очереди PHD08 ниже. Разные очереди между собой GPU не координируют.

| Датасет | Серии | Что делает очередь |
|---|---|---|
| PHD08 | `p1,p2,p3,p4,p5` по отдельности | P1/P2: известные классы; P3: retrieval; P4: оценка checkpoint P3; P5: HangulPrinted → PHD08 |
| Omniglot | `o1,o2,o3` можно объединить; `o4` и `o5` отдельно | Closed-set full/5/10, retrieval, few-shot и отдельная run-folder one-shot серия |
| CUB | `c1,c2,c3` можно объединить; `c4` отдельно | Retrieval, closed-set full/5/10 и few-shot |
| CLINC150 | `l1,l2,l3,l5,l6` можно объединить; `l4` отдельно | Closed-set full/5/10, retrieval, open-set и замороженный MiniLM |

Для Omniglot/CUB/CLINC реализованная базовая матрица включает четыре варианта
CATML, три варианта Triplet и Cross-entropy. Closed-set конфиги метрического
обучения и retrieval-вариант Cross-entropy генерируются из конфигов проекта;
их нужно сначала проверить пилотом. Выбор checkpoint для closed-set ведётся по
accuracy ближайшего эталона; в Cross-entropy дополнительно сохраняется accuracy
классификационной головы. `l5` обучает отдельные модели для open-set и калибрует
порог на validation; `l6` оценивает замороженный MiniLM без обучения.

Например, один seed по всем основным обучающим сериям Omniglot — 32 задания:

```bash
python scripts/run_experiment_batch.py \
  --series o1,o2,o3 --phase development \
  --data-root /путь/к/omniglot --models-dir "$MODEL_DIR" \
  --seeds 1 --gpus 0 --workers-per-gpu 2
```

Для CUB используй `--series c1,c2,c3`; доступен `--backbone resnet50` или `vit`.
Для CLINC — `--series l1,l2,l3,l5,l6`, `--backbone roberta` или `minilm`;
`l6` всегда использует замороженный MiniLM. Готовность pretrained-весов и бюджет
конфигов нужно проверить на вычислительной машине. При `--phase final` включается
test; в retrieval итоговая модель обучается на train+validation классах.

Few-shot серии используют завершённые retrieval-модели той же фазы:

```bash
python scripts/run_experiment_batch.py \
  --series o4 --phase development \
  --data-root /путь/к/omniglot --models-dir "$MODEL_DIR" \
  --source-batch experiments/jobs/omniglot_batch_o1_o2_o3_development_resnet18 \
  --gpus 0
```

Аналогично C4 берёт C1, L4 — L3, P4 — P3. Для final укажи завершённую финальную
очередь и `--phase final`. Проверка CUB few-shot validation делит официальные
train-изображения класса на пять query и остальные support, без пересечения;
test использует официальные test-изображения. O5 запускается только в final и
ожидает `all_runs` (или явный `--oneshot-root`); соответствие этих заданий точному
официальному протоколу ещё не подтверждено.

В новой матрице обе APM-строки используют одинаковый `apm_gamma=1` по умолчанию,
чтобы сравнение с/без Auto-clustering не меняло одновременно степень APM. Ранее
сформированные очереди не изменяются и могут иметь `apm_gamma=2` для `apm_auto`;
не смешивай их без проверки конфига и колонки CSV. Для новой версии используй
новый `--output-dir`.

Полный текст исследовательских планов также включает PFML, Proxy-AN, обучаемый
ProtoNet, расширенные селекторы и другие абляции, которых в этой очереди нет.
Необучаемый ближайший прототип не является обучаемым ProtoNet.

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

## Parquet-каталоги с Hugging Face

Чтобы создать Parquet-версии исходных данных, выполни команды из
[README экспериментов](README.md#parquet-копия-для-hugging-face). Загрузчик
поддерживает оба варианта: исходные папки с PNG/JSON и локально скачанный
Parquet-каталог с `dataset_manifest.json`. Скачивай с Hub весь каталог, включая
этот манифест и `alphabet.json` KorSynthetic.

Для P1–P4 вместо стандартного корня передай путь к PHD08 Parquet-каталогу:

```bash
python scripts/launch_experiment.py \
  --config phd08_smoke_pilot \
  --models-dir "$MODEL_DIR" \
  --data-root /mnt/datasets/phd08
```

Для P5 укажи оба локальных каталога:

```bash
python scripts/launch_experiment.py \
  --config phd08_synthetic_metric_2350 \
  --cross-entropy-config phd08_synthetic_2350 \
  --phd08-baseline --models-dir "$MODEL_DIR" \
  --data-root /mnt/datasets/kor_synthetic \
  --eval-data-root /mnt/datasets/phd08 \
  --seeds 1,2,3,4,5 --run-test \
  --output-dir experiments/jobs/phd08_p5_2350_parquet
```

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
В каждом `last.pt` и `best.pt` есть раздел `ideals`: центры классов за эпоху,
порядок которых задан списком `ideals.class_labels`, и число обучающих примеров
на каждый центр в `ideals.counts`.
Задания серии выполняются последовательно; первая ошибка останавливает очередь.

## Параллельные запуски и общий CSV

Разные команды launcher можно запустить в разных терминалах. Автоматически
создаваемые каталоги включают микросекунды и ID процесса; если задаёшь `--output` или
`--output-dir` вручную, используй отдельный путь для каждого запуска. Не запускай
две команды одновременно в одной папке результатов и не возобновляй один и тот
же checkpoint в двух процессах.

По умолчанию конфиги используют GPU 0. Если нужны разные видеокарты, например,
запускай команды с `CUDA_VISIBLE_DEVICES=0` и `CUDA_VISIBLE_DEVICES=1`
соответственно: внутри каждого процесса выбранная видеокарта будет устройством 0.
Проверь доступную память на общих GPU. Параллельная нагрузка влияет на измерения
времени, поэтому сравнивать скорость методов лучше при одинаковых условиях.

Метрики каждой эпохи, validation и test сохраняются в папке своего запуска.
Чтобы автоматически обновлять общую таблицу по мере работы, запусти в отдельном
терминале из корня проекта:

```bash
python scripts/collect_results.py --watch
```

Каждую минуту собирается `experiments/results.csv`; остановка сборщика через
Ctrl+C не останавливает обучение. CSV можно переслать и во время обучения, и после.
Он также включает результаты предыдущих запусков. Для разового обновления без
фонового наблюдения запусти `python scripts/collect_results.py`. Пути вне
`experiments/runs/` и `experiments/jobs/` добавляются через `--search-root /путь`.

В таблице одна строка на эпоху и отдельные строки для итоговой или самостоятельной
оценки. Поля `source`, `record_type`, `split`, параметры запуска и метрики с
префиксами `train.`, `validation.`, `test.` позволяют отличать результаты.
Пустая validation означает, что она в эту эпоху не выполнялась. CSV собирается из
сохранённых файлов и не определяет причину остановки или ошибки: консоль и исходные
файлы запуска остаются источником диагностики. Таблица заменяется целиком атомарно,
поэтому обучение в нескольких терминалах не пишет в общий файл одновременно.

## Границы реализованной части планов

Текущая основная PHD08-матрица соответствует восьми сравнениям из плана. Более
широкие baseline-строки и абляции из раздела 6 плана (PFML, Proxy-AN,
обучаемый ProtoNet и другие селекторы) пока не добавлены. PHD08 split исключает
пересечение восстановленных по именам файлов групп; font-disjoint проверка не
проводится. Точный официальный протокол Omniglot O5 также не подтверждён. Не
используй результаты итоговой оценки для подбора гиперпараметров.

## Матрицы PHD08

### Пять параллельных пачек

`scripts/run_phd08_batch.py` запускает независимые задания матрицы параллельно,
назначая каждому GPU через `CUDA_VISIBLE_DEVICES`. Пачки `p1`, `p2`, `p3`, `p4`
и `p5` соответствуют постановкам плана; P5 включает оба алфавита. По умолчанию
используются пять seeds и одна задача на GPU. Выбери GPU и число параллельных
задач исходя из доступной памяти и CPU/диска. Например, одна GPU и два процесса:

```bash
python scripts/run_phd08_batch.py \
  --batch p3 --phase development \
  --phd-root /путь/к/PHD08 --models-dir /путь/к/моделям \
  --seeds 1 --epochs 50 --gpus 0 --workers-per-gpu 2
```

Это восемь сравнений на validation; `--epochs` меняет бюджет для всех восьми.
`--dry-run` создаёт план и показывает число заданий без обучения. Повтори ту же
команду без `--dry-run`, чтобы выполнить план. Для двух GPU используй `--gpus 0,1`;
при `--workers-per-gpu 1` будет не больше одной задачи на каждой карте.

`--phase development` не запускает test. После фиксации настроек по validation
используй `--phase final`: для P3 это переобучение на train+validation классах и
тестирование на отложенных классах, для P1/P2/P5 — обучение с validation и test.
P1, P2, P3 и P5 с пятью seeds содержат соответственно 40, 80, 40 и 80 обучений.
После завершения финальной пачки P3 создай P4:

```bash
python scripts/run_phd08_batch.py \
  --batch p4 --phase final \
  --phd-root /путь/к/PHD08 --models-dir /путь/к/моделям \
  --p3-dir experiments/jobs/phd08_batch_p3_final --gpus 0,1
```

P4 оценивает каждый сохранённый P3 checkpoint в 1-shot и 5-shot; отдельного
обучения нет. Для P5 дополнительно укажи `--kor-root /путь/к/KorSynthetic`.

Каталог пачки по умолчанию — `experiments/jobs/phd08_batch_<серия>_<фаза>`.
Результаты и логи задач находятся в `runs/<задача>/attempt_<номер>/`, статус — в
`batch_state.json`. `experiments/results.csv` автоматически обновляется после
каждой завершённой задачи; он также содержит статусы и последние строки ошибок.
Для состояния очереди: `python scripts/run_phd08_batch.py --batch p3 --status`
(добавь `--phase final` для итоговой пачки).

Повтор той же команды продолжает незавершённую очередь, сохраняя успешные задачи.
После ошибки новые задачи не выдаются, уже работающие завершаются; после исправления
добавь `--retry-failed`. Прерванная или неудачная задача начинается заново в новой
папке попытки, прежние логи и checkpoint сохраняются. Для изменения набора seeds,
бюджета или данных используй новый `--output-dir`: существующая очередь не меняет
параметры. `Ctrl+C` останавливает процессы текущей пачки.

### Сначала: короткий smoke-запуск

Сначала проверь команду без обучения, затем выполни один короткий проход:

```bash
python scripts/launch_experiment.py \
  --config phd08_smoke_pilot \
  --models-dir "$MODEL_DIR" \
  --dry-run

python scripts/launch_experiment.py \
  --config phd08_smoke_pilot \
  --models-dir "$MODEL_DIR" \
  --output experiments/runs/phd08_smoke_pilot
```

Пилот делает два шага CATML-обучения и nearest-exemplar validation на пяти
классах PHD08. Он проверяет запуск модели, загрузку PNG, обучение, checkpoint и
метрики; его результат не входит в PhD-таблицы. При успехе переходи к P1/P2.

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

# Эксперименты

Здесь хранятся канонические конфиги и манифесты разбиений CLINC150,
CUB-200-2011, Omniglot, PHD08 и KorSynthetic. Одиночный запуск выполняется через
корневой `train.py`; матрицы и очереди — через `scripts/run_experiment_batch.py`.
Конфиги содержат единственную поддерживаемую схему. Реализации разнесены по
`experiment_data.py`, `experiment_losses.py`, `experiment_models.py`,
`experiment_sampling.py`, `experiment_metrics.py` и `experiment_train.py`.

Порядок запусков, соответствие конфигов исследовательским планам и готовые
команды приведены в [руководстве по запуску](RUNBOOK.md).

## Окружение для Blackwell

Начальная конфигурация окружения: Python 3.13.6, PyTorch 2.14.1, TorchVision
0.29.1 и стабильные пакеты CUDA 13.0. Все зависимости устанавливаются из одного
файла:

```bash
python3.13 --version  # ожидается Python 3.13.6
python3.13 -m venv .venv-blackwell
source .venv-blackwell/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Для изолированной машины на совместимом компьютере с интернетом сначала скачай
все пакеты и их зависимости, затем перенеси каталог `wheelhouse` и установи их
без доступа к сети:

```bash
python3.13 -m pip download --only-binary=:all: --dest wheelhouse -r requirements.txt
python -m pip install --no-index --find-links /путь/к/wheelhouse -r requirements.txt
```

Для CLINC MiniLM загружается через Transformers; пакет `sentence-transformers`
для экспериментальных конфигов не требуется.

CUDA runtime входит в зависимости PyTorch, поэтому отдельно устанавливать CUDA
Toolkit нужно только для сборки собственных расширений CUDA. См. [пакеты PyTorch
2.14.1 и TorchVision 0.29.1](https://pytorch.org/get-started/previous-versions/) и
[требования NVIDIA к драйверу CUDA 13.0](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/).

Перед долгим запуском проверь вычислительную машину:

```bash
nvidia-smi
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**30, torch.cuda.is_bf16_supported())'
```

Конфиги используют BF16, закреплённую память, воспроизводимую инициализацию
рабочих процессов по эпохам и формат каналов channels-last для изображений.
Постоянные рабочие процессы выключены по умолчанию, поскольку сохраняют состояние
генератора случайных чисел между эпохами и при продолжении запуска. Начальные
размеры пакетов P×K не гарантируют максимальную загрузку видеокарты с 204 ГБ
памяти. Увеличивай `classes_per_batch`, `samples_per_class` и размер пакета
оценки только после измерения скорости и проверки, что сравниваемые методы
используют одинаковые настройки. `torch.compile` пока не включён: время компиляции
и специализация графа не должны менять исходное сравнение времени работы.

Для запуска с готовыми весами передавай launcher-у `--models-dir /путь/к/моделям`.
Репозитории Hugging Face ищутся по их идентификаторам, например
`FacebookAI/roberta-base` и `sentence-transformers/all-MiniLM-L6-v2`. Веса
TorchVision ищутся в подпапке `torchvision/checkpoints/`. Если этот параметр
передан, используются только локальные файлы; автоматической загрузки из сети
не будет. Ожидаемая структура каталогов показана в [руководстве](RUNBOOK.md).

При поиске галерея векторов хранится на GPU, а оценки запросов вычисляются
частями. При заданных размерах векторов это подходит для галереи PHD08 в
протоколе поиска на новых классах. Если другой набор данных или модель не
помещаются в память, сначала уменьши размер пакета оценки; перенос вычислений на
CPU или хранение в отображаемом в память файле (`memmap`) нужно отдельно
зафиксировать как режим оценки.

## Манифесты данных

Манифесты для наборов данных из `/home/arseniy/data` сохранены в
`experiments/manifests/`. Пересоздавай манифест после изменения данных или
начального значения генератора `seed`:

```bash
python scripts/prepare_manifests.py --dataset cub200 --root /home/arseniy/data/CUB-200-2011 --output experiments/manifests/cub200.json --seed 20261113
python scripts/prepare_manifests.py --dataset omniglot --root /home/arseniy/data/omniglot --output experiments/manifests/omniglot.json --seed 20261113
python scripts/prepare_manifests.py --dataset phd08 --root /home/arseniy/data/PHD08_dataset --kor-root /home/arseniy/data/kor_synthetic --output experiments/manifests/phd08.json --seed 20261113
python scripts/prepare_manifests.py --dataset clinc150 --root /home/arseniy/data/clinc150 --output experiments/manifests/clinc150.json --seed 20261113
python scripts/prepare_manifests.py --dataset kor_synthetic --root /home/arseniy/data/kor_synthetic --phd-root /home/arseniy/data/PHD08_dataset --output experiments/manifests/kor_synthetic.json --seed 20261113
```

## Parquet-копия для Hugging Face

`scripts/convert_datasets_to_parquet.py` создаёт локальный Dataset-репозиторий:
строки PHD08 хранят встроенные PNG и ID семейства. Строки KorSynthetic хранят
исходный JSON целиком в `source_json` для байт-в-байт восстановления и отдельные
float32-массивы фона и маски для обучения. Классы обрабатываются
параллельно; по умолчанию используется не более восьми worker-ов. На медленном
сетевом диске уменьши `--workers`. Для PHD08 обязательно укажи файл алфавита через
`--alphabet-json`: конвертер записывает символ хангыль, fingerprint алфавита
и копию JSON в Parquet-каталог.

В Parquet-версии PHD08 изображения доступны в колонке `image`. Для KorSynthetic
числовые массивы по умолчанию не превращаются в изображения. Если нужен просмотр
строк прямо в Dataset Viewer Hugging Face, добавь KorSynthetic-конвертации флаг
`--preview-images`. Он добавляет отдельный статический preview, созданный из
образца; обучение продолжает пользоваться исходными массивами и случайной
сборкой фона. Превью увеличивает время конвертации и размер Parquet; без флага
экспортируются только исходные массивы.

```bash
python scripts/convert_datasets_to_parquet.py \
  --dataset kor_synthetic \
  --root /home/arseniy/data/kor_synthetic \
  --output /mnt/datasets/kor_synthetic \
  --workers 8

python scripts/convert_datasets_to_parquet.py \
  --dataset phd08 \
  --root /home/arseniy/data/PHD08_dataset \
  --alphabet-json /home/arseniy/data/kor_synthetic/alphabet.json \
  --output /mnt/datasets/phd08 \
  --workers 8
```

Конвертация выполняется по классам и продолжается с `--resume` после прерывания.
Сначала можно проверить формат на двух классах в отдельных временных папках,
добавив `--class-ids 18,19`. Частичный набор годится только для smoke-проверки:
полные экспериментальные конфиги отклонят его по контрольным суммам манифеста.
Полный локальный Parquet-каталог передавай через `--data-root`; для P5 укажи
`--data-root` KorSynthetic и `--eval-data-root` PHD08.

Выходная папка содержит `dataset_manifest.json`, Parquet-шарды и копию алфавита
KorSynthetic. Её можно загрузить как Dataset-репозиторий. Перед публичной
публикацией добавь проверенный источник и лицензию в README датасета.

Чтобы восстановить привычные папки из скачанного Parquet-каталога, используй
`scripts/restore_datasets_from_parquet.py`:

```bash
python scripts/restore_datasets_from_parquet.py \
  --dataset kor_synthetic \
  --parquet-root /mnt/datasets/kor_synthetic \
  --output /mnt/restored/kor_synthetic \
  --workers 8

python scripts/restore_datasets_from_parquet.py \
  --dataset phd08 \
  --parquet-root /mnt/datasets/phd08 \
  --output /mnt/restored/PHD08_dataset \
  --workers 8
```

PHD08 PNG-файлы возвращаются байт-в-байт. Parquet v2 для KorSynthetic хранит
исходные JSON-байты в `source_json`, поэтому обратная конвертация восстанавливает
их байт-в-байт. Более старый Parquet v1 содержит только разобранные массивы и не
может вернуть исходные JSON-байты. Для продолжения прерванного восстановления
добавь `--resume`.

Манифест CUB следует `images.txt` и указывает восемь дополнительных JPG-файлов,
которых нет в метаданных. Повторяющиеся тексты CLINC сохраняются в официальных
разбиениях; эта политика зафиксирована в манифесте. Протокол KorSynthetic для
корейских слогов содержит 11 172 символа и исключает 18 не-хангыльских символов
из предоставленного алфавита. По именам файлов PHD08 строится устойчивый ID
группы: папка класса и первые четыре поля имени файла. Полная проверка показала
в каждом классе 243 такие группы по 9 вариантов; последние два поля задают
комбинации двух трёхзначных параметров. P1/P2 делят группы 60/20/20, а P3/P4
делят группы между галереей и запросами внутри отложенных классов. Это
разбиение по семействам, восстановленным из структуры имён, но оно не является
разбиением по шрифтам: шрифт встречается во всех частях.

В сохранённом `phd08.json` выполнена полная проверка количества файлов: во всех
2 350 классах ровно 2 187 PNG, всего 5 139 450 изображений. При обычной
перегенерации манифеста полный проход по 5,1 млн файлов выключен; чтобы повторить
проверку, добавь `--deep-scan`.

В PHD08 и KorSynthetic номер папки `N` напрямую указывает на элемент `N` в
`alphabet.json` или `kor_101_mfcr.json`. Таблица соответствий не нужна. Генератор манифеста проверяет,
что все 2 350 номеров папок PHD08 соответствуют разным символам хангыль и что
папки с такими же номерами есть в KorSynthetic. Для исходного KorSynthetic
поддерживаются `alphabet.json` и классический `kor_101_mfcr.json`; папки классов
могут лежать в `data/` или `retrieve_mtr_per_class/`. При оценке P5 тот же индексный
способ применяется во время запуска; проверяются контрольные суммы алфавита и
набора меток классов.

## Запуск

Одиночный запуск использует корневой `train.py`, явный путь к config и явную
выходную папку:

```bash
python train.py \
  --config experiments/configs/phd08_metric_dev.json \
  --data-root /путь/к/PHD08_dataset \
  --models-dir /путь/к/моделям \
  --output experiments/runs/phd08_metric_dev/seed1
```

Добавь `--run-test` только для заранее выбранного test-протокола. Для
предобученной модели используй `--evaluate-pretrained --split validation` или
`--split test`; для open-set test добавь `--calibrate-with-validation`.

Матрицы PHD08, Omniglot, CUB-200-2011 и CLINC150 запускаются через
`scripts/run_experiment_batch.py`; параметры, коды серий и порядок P1–P5 указаны
в [RUNBOOK.md](RUNBOOK.md). Для проверки плана перед запуском используй
`--dry-run`. Batch runner создаёт уникальные конфиги и папки запусков.

P5 использует два набора данных. Для него `--data-root` задаёт KorSynthetic,
`--eval-data-root` — PHD08; для серии PHD08 в batch runner эти корни передаются
как `--kor-root` и `--data-root` соответственно. Финальная матрица запускается
с `--phase final`.

`kornet2` загружает архитектуру из `experiments/models/kornet2.json`. В CLINC
используются RoBERTa-base и MiniLM с закреплёнными ревизиями. CUB использует
ResNet-50 или ViT-B/16 с зафиксированными весами ImageNet. При передаче
`--models-dir` веса должны присутствовать локально; загрузка из сети не
используется.

## Определение CATML в текущей реализации

`CATMLLoss` из `experiment_losses.py` соответствует формуле из
`/home/arseniy/phd/Dissertation/catml_definition.tex`:

- `h` — выход энкодера; для расстояний триплета применяется покоординатный
  `softplus(h)`.
- Положительный штраф использует обычное, не возведённое в квадрат расстояние;
  штраф ранжирования задаётся через `softplus(d_ap - d_an + margin)`.
- Штраф радиуса класса вычисляется по `h`, а не по представлению после softplus.
- По умолчанию значения усредняются по триплетам и взвешиваются по примерам
  класса `m_c / B`. В P×K режиме центры текущего пакета дифференцируемы; в
  режиме `global` после warm-up радиусный член использует общий snapshot-центр,
  который фиксирован на время backward-прохода.

Способ усреднения и режим градиента центров задаются параметрами конфига и
сохраняются в checkpoint. Они являются частью протокола и должны совпадать между
сравниваемыми строками. Ненулевой `norm_epsilon` задаёт сглаженную норму.

APM обновляет вероятности выбора классов по радиусам, наблюдаемым в обучающих
пакетах. Небольшая минимальная вероятность для ещё не наблюдавшихся классов
не даёт им навсегда исчезнуть из выборки; при нулевом значении такие классы могут
перестать выбираться. Auto-clustering использует векторы обучающих примеров и
выбирает соседний класс из текущего class graph с заданной вероятностью; если
подходящего соседа нет, применяется случайный выбор. Класс graph обновляется
только по обучающим embeddings.

Для triplet-задач доступны два режима генерации. `sampler.batch_mode="pk"`
использует классические `classes_per_batch` × `samples_per_class` кандидаты.
`sampler.batch_mode="global"` напрямую формирует `training.batch_count`
логических батчей по `training.batch_size` триплетов. Каждый логический батч разбивается на
`training.minibatch_size` триплетов, градиенты накапливаются и оптимизатор
обновляется один раз на логический батч.
`batch_mode="global"` задаёт способ сборки батча, а `negative="global_hard"` —
конкретное правило выбора отрицательного примера.

В режиме `global` anchor выбирается равномерно по классам либо через APM,
positive — случайно из того же класса. Negative можно выбрать случайно из всех
train-классов, из class graph либо с помощью `global_hard` / `semi_hard`. Для
global hard/semi-hard и для APM после появления общего centroid snapshot перед
эпохой считаются эмбеддинги всего train-разбиения; hard mining выполняет точный
поиск по всем train-примерам.
Используются детерминированные inference-преобразования; размер GPU-пачки этого
прохода задаёт `training.mining_batch_size` (по умолчанию
`evaluation.batch_size`). Поиск не ограничен текущим батчем, но его стоимость
растёт как число anchors × число train-примеров. Матрица эмбеддингов и индекс
SampleRef хранятся в RAM; изображения загружаются порциями при обучении. В
отличие от `batch_hard` в P×K режиме, это глобальный поиск; `batch_hard` в
режиме `global` не разрешён.

В текущей реализации P×K режима `batch_hard` выбирает ближайший negative;
positive остаётся случайным.

Центры для режима `global` накапливаются по всем обучающим microbatch-ам. Раз в
`sampler.make_cluster_on_epoch` эпох из накопителей публикуется общий snapshot;
по нему обновляется class graph, а накопители обнуляются. Следующий интервал
использует этот snapshot и для Auto-clustering, и для радиусов APM, и для
радиусного члена CATML. В начале каждой эпохи APM считает радиусы train-примеров
относительно активного snapshot и обновляет вероятности классов. До первого
snapshot APM остаётся равномерным, CATML использует microbatch-центры. После
публикации snapshot CATML считает радиусный член относительно него; градиент
проходит через эмбеддинги, но не через фиксированные snapshot-центры.

Каждое обновление записывается отдельным JSON в `sampler_updates/`: событие
`class_graph` содержит словарь `класс → участники кластера` и центроиды, событие
`apm_probabilities` — словарь `класс → вероятность`. В режиме `global` также
сохраняется `centroid_snapshot`, если обновился snapshot без Auto-clustering.
После обучения сохраняется обзорный `<run>/training_curves.png` и отдельные
графики loss-компонентов, validation-метрик, распределения APM и структуры
кластеров — если соответствующие данные есть в запуске.

## Ограничения, которые нужно учитывать

- PHD08-группы восстановлены из имён файлов и подтверждены полной проверкой
  структуры 2 187 файлов в каждом классе. Это семейственно-групповое разбиение,
  а не font-disjoint протокол; выводы о переносе на новые шрифты из него не
  следуют.
- Эпизоды N-way/K-shot (N классов, K опорных примеров) формируются
  воспроизводимо по начальному значению генератора из манифеста. Точные
  идентификаторы опорных примеров и запросов сохраняются в
  `episodes_validation.json` и `episodes_test.json`, чтобы разные методы
  использовали одинаковые эпизоды. Порог CLINC для неизвестных намерений
  настраивается на проверочной части и применяется к итоговой оценке.
- Локальная папка Omniglot `all_runs` задаёт отдельный run-folder one-shot
  протокол; использовать его как официальный benchmark можно только после
  сверки содержимого папок и правил оценки.
- PFML, Proxy-AN, обучаемый ProtoNet и дополнительные baseline-строки расширенной
  матрицы плана пока не входят в восьмистрочную основную матрицу.
- Каждый конфиг содержит стартовый, относительно небольшой бюджет обучения.
  Полную матрицу запусков следует определять после пилота и измерения скорости
  на целевой вычислительной машине.

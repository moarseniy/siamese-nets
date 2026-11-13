# Как запускать эксперименты

Этот файл связывает планы `EXPERIMENT_PLAN_*.md` с конфигами и командами.
Команды запускайте из корня репозитория на машине с CUDA; корни данных ниже
соответствуют указанным тобой папкам.

## 1. Сначала один пилот

Установи окружение по разделу Blackwell в [README экспериментов](README.md),
активируй `.venv-blackwell` и проверь `nvidia-smi` и `torch.cuda.is_available()`.
Потом создай короткий пилот CUB: один epoch, два train-step, validation на
заданном dev-разбиении.

```bash
python scripts/run_matrix.py \
  --config experiments/configs/cub200_metric_dev.json \
  --data-root /home/arseniy/data/CUB-200-2011 \
  --output-dir experiments/jobs/pilot_cub \
  --seeds 1 --losses catml --selectors random \
  --examples-per-class 10 --epochs 1 --steps-per-epoch 2
bash experiments/jobs/pilot_cub/run.sh
```

`run_matrix.py` только создаёт конфиг и команду. Обучение начинается на
последней строке при запуске `run.sh`. Команды в `run.sh` идут последовательно
и останавливаются при первой ошибке. Этот пилот всё ещё выполняет validation;
он не запускает test.

## 2. Запуск одного конфига

Обычный запуск обучает модель и считает настроенную validation-метрику.
Например, CLINC closed-set:

```bash
python train.py \
  --config experiments/configs/clinc150_classification.json \
  --data-root /home/arseniy/data/clinc150 \
  --output experiments/runs/clinc150/classification
```

У каждого конфига есть фиксированный стартовый бюджет. Путь в `--output`
должен быть отдельным для каждого запуска: там сохраняются `resolved_config.json`,
manifest, `metrics.jsonl`, `best.pt` и `last.pt`. Для продолжения прерванного
запуска передай `--resume <путь-к-папке>/last.pt`, сохранив тот же `--output`.

`--run-test` включает test из конфига после обучения. Используй его только
когда выбор модели и настройки уже зафиксированы. Retrieval-конфиги с суффиксом
`metric_dev` намеренно не содержат test: для них ниже приведена финальная
процедура через `--stage final`.

## 3. Какие конфиги соответствуют планам

Для всех обычных запусков команда одна и та же:

```bash
python train.py --config experiments/configs/<имя>.json \
  --data-root <корень-данных> --output experiments/runs/<имя>/<seed>
```

В таблице указаны имя конфига и корень данных. Передавай `--run-test` только
для зафиксированного итогового запуска, если у конфига настроен test.

| План | Конфиг(и) | `--data-root` | Что запускается |
|---|---|---|---|
| CLINC L1 | `clinc150_classification.json` | `/home/arseniy/data/clinc150` | Closed-set, 150 известных намерений |
| CLINC L2 | `clinc150_classification.json` | то же | Те же классы, бюджеты `all`, `5`, `10` задаются матрицей ниже |
| CLINC L3 | `clinc150_metric_dev.json` | то же | Dev retrieval; финальный test через `--stage final` |
| CLINC L4 | `clinc150_fewshot_dev.json` | то же | Few-shot новых намерений; по умолчанию 5-way 1-shot, `--k-shot 5` для 5-shot |
| CLINC L5 | `clinc150_open_set.json` | то же | Known/OOS; порог калибруется на validation |
| CLINC encoder baseline | `clinc150_sentence_transformer_frozen.json`, `clinc150_sentence_transformer_finetune_dev.json` | то же | Замороженный и дообучаемый MiniLM; команды ниже |
| CUB C1 | `cub200_metric_dev.json` | `/home/arseniy/data/CUB-200-2011` | Dev retrieval новых видов; финальный test через `--stage final` |
| CUB C2/C3 | `cub200_classification.json` | то же | Closed-set и бюджеты `all`, `5`, `10` |
| CUB C4 | `cub200_fewshot_dev.json` | то же | Few-shot новых видов; `--k-shot 1` или `5` |
| CUB backbone comparison | `cub200_vitb16_metric_dev.json` | то же | ViT-B/16 dev retrieval; финальный test через `--stage final` |
| Omniglot O1/O2 | `omniglot_classification.json` | `/home/arseniy/data/omniglot` | Closed-set и бюджеты `all`, `5`, `10` |
| Omniglot O3 | `omniglot_metric_dev.json` | то же | Alphabet-disjoint dev retrieval; финальный test через `--stage final` |
| Omniglot O4 | `omniglot_fewshot_dev.json` | то же | Few-shot новых алфавитов; `--k-shot 1` или `5` |
| Omniglot O5 | `omniglot_run_folder_oneshot.json` | то же | Обучение и оценка по `all_runs`; запускать с `--run-test` |
| PHD08 P1/P2 | `phd08_classification.json` | `/home/arseniy/data/PHD08_dataset` | Closed-set и бюджеты `all`, `5`, `10`; текущий split не source-disjoint |
| PHD08 P3 | `phd08_metric_dev.json` | то же | Dev class-disjoint retrieval; финальный test через `--stage final` |
| PHD08 P4 | `phd08_fewshot_dev.json` | то же | Few-shot новых классов; `--k-shot 1` или `5` |
| PHD08 P5 | `phd08_synthetic_2350.json`, `phd08_synthetic_full.json` | `/home/arseniy/data/kor_synthetic` | Синтетическое обучение → PHD08; дополнительно `--eval-data-root /home/arseniy/data/PHD08_dataset` |

Примеры специальных запусков:

```bash
# Замороженный MiniLM: оценка на validation без обучения
python train.py \
  --config experiments/configs/clinc150_sentence_transformer_frozen.json \
  --data-root /home/arseniy/data/clinc150 \
  --evaluate-pretrained --split validation \
  --output experiments/runs/clinc150/st_frozen/validation.json

# После фиксации протокола: отдельная оценка на test
python train.py \
  --config experiments/configs/clinc150_sentence_transformer_frozen.json \
  --data-root /home/arseniy/data/clinc150 \
  --evaluate-pretrained --split test \
  --output experiments/runs/clinc150/st_frozen/test.json

# Дообучаемый MiniLM: обычный train + validation
python train.py \
  --config experiments/configs/clinc150_sentence_transformer_finetune_dev.json \
  --data-root /home/arseniy/data/clinc150 \
  --output experiments/runs/clinc150/st_finetune/seed1

# P5: тест на PHD08 включается явно; тестовый split использует все 2,350 классов
python train.py \
  --config experiments/configs/phd08_synthetic_2350.json \
  --data-root /home/arseniy/data/kor_synthetic \
  --eval-data-root /home/arseniy/data/PHD08_dataset \
  --output experiments/runs/p5/kor2350_to_phd08/seed1 \
  --run-test
```

У frozen MiniLM итоговую retrieval-оценку запускай отдельно с `--split test`
после фиксации протокола. Флаг `--calibrate-with-validation` для retrieval не
нужен; он предназначен для задач с порогом open-set.

Для финального retrieval fine-tuned MiniLM применим final-stage рецепт из
предыдущего раздела, передав `clinc150_sentence_transformer_finetune_dev.json`
вместо `clinc150_metric_dev.json`.

Для O5 конфиг использует путь
`/home/arseniy/data/omniglot/python/all_runs`. Проверь его наличие на кластерном
узле перед запуском. Локальная папка пока не подтверждена как точное официальное
разбиение заданий Omniglot.

## 4. Матрицы запусков

### Closed-set: все данные, 5 и 10 примеров на класс

Эта функция создаёт по 15 job-ов на датасет: три бюджета × пять seed. Она
использует cross-entropy конфиг и случайный sampler.

```bash
make_closed_set_jobs() {
  local config_path="$1" data_root="$2" output_dir="$3"
  python scripts/run_matrix.py \
    --config "$config_path" --data-root "$data_root" --output-dir "$output_dir" \
    --seeds 1,2,3,4,5 --losses cross_entropy --selectors random \
    --examples-per-class all,5,10
}

make_closed_set_jobs experiments/configs/clinc150_classification.json \
  /home/arseniy/data/clinc150 experiments/jobs/clinc_closed
make_closed_set_jobs experiments/configs/cub200_classification.json \
  /home/arseniy/data/CUB-200-2011 experiments/jobs/cub_closed
make_closed_set_jobs experiments/configs/omniglot_classification.json \
  /home/arseniy/data/omniglot experiments/jobs/omniglot_closed
make_closed_set_jobs experiments/configs/phd08_classification.json \
  /home/arseniy/data/PHD08_dataset experiments/jobs/phd08_closed
```

После фиксации настроек добавь `--run-test` в функцию при генерации итоговых
job-ов. Этот флаг попадёт в каждую созданную команду `train.py`. Для запуска
созданных матриц выполни `bash <output-dir>/run.sh` для каждой папки. PHD08
результаты при текущем manifest нельзя описывать как
source/font-disjoint: в данных не обнаружены идентификаторы источника.

Когда все четыре папки сгенерированы, запустить весь closed-set блок можно так:

```bash
for job_script in experiments/jobs/{clinc,cub,omniglot,phd08}_closed/run.sh; do
  bash "$job_script"
done
```

### Retrieval: CATML и triplet с нужными селекторами

Следующая функция создаёт 35 dev job-ов для датасета: пять seed на каждую из
четырёх CATML-схем и трёх triplet-схем. Сначала результаты выбираются по
validation; test они не запускают.

```bash
make_retrieval_dev_jobs() {
  local config_path="$1" data_root="$2" output_dir="$3"
  python scripts/run_matrix.py \
    --config "$config_path" --data-root "$data_root" \
    --output-dir "$output_dir/catml" --seeds 1,2,3,4,5 \
    --losses catml --selectors random,apm,auto_clusters,apm_auto \
    --examples-per-class all
  python scripts/run_matrix.py \
    --config "$config_path" --data-root "$data_root" \
    --output-dir "$output_dir/triplet" --seeds 1,2,3,4,5 \
    --losses triplet --selectors random,semi_hard,batch_hard \
    --examples-per-class all
}

make_retrieval_dev_jobs experiments/configs/clinc150_metric_dev.json \
  /home/arseniy/data/clinc150 experiments/jobs/clinc_retrieval
make_retrieval_dev_jobs experiments/configs/cub200_metric_dev.json \
  /home/arseniy/data/CUB-200-2011 experiments/jobs/cub_retrieval
make_retrieval_dev_jobs experiments/configs/omniglot_metric_dev.json \
  /home/arseniy/data/omniglot experiments/jobs/omniglot_retrieval
make_retrieval_dev_jobs experiments/configs/phd08_metric_dev.json \
  /home/arseniy/data/PHD08_dataset experiments/jobs/phd08_retrieval
```

Затем запусти созданные `run.sh` из каждой `catml` и `triplet` папки. После
выбора метода и параметров создай финальные test job-ы для выбранных запусков:

```bash
python scripts/run_matrix.py \
  --config experiments/configs/phd08_metric_dev.json \
  --data-root /home/arseniy/data/PHD08_dataset \
  --output-dir experiments/jobs/phd08_final_catml_random \
  --stage final --seeds 1,2,3,4,5 --losses catml --selectors random \
  --examples-per-class all --run-test
bash experiments/jobs/phd08_final_catml_random/run.sh
```

Для CLINC, CUB и Omniglot используй ту же команду, заменив конфиг, `--data-root`
и `--output-dir`. `--stage final` доступен только для retrieval dev-конфигов:
он переобучает модель на train + validation классах и один раз оценивает на
отложенных test-классах.

После генерации всех четырёх пар dev-матриц можно запустить их последовательно:

```bash
for job_script in \
  experiments/jobs/{clinc,cub,omniglot,phd08}_retrieval/{catml,triplet}/run.sh; do
  bash "$job_script"
done
```

### Few-shot и open-set

Для 1-shot/5-shot `run_matrix.py` меняет K одновременно в validation и test.
Функция ниже создаёт пять seed для одного dataset и заданного K; повтори её для
каждой строки таблицы, если нужны все четыре датасета и оба значения K.

```bash
make_fewshot_jobs() {
  local config_path="$1" data_root="$2" output_dir="$3" k_shot="$4"
  python scripts/run_matrix.py \
    --config "$config_path" --data-root "$data_root" --output-dir "$output_dir" \
    --seeds 1,2,3,4,5 --losses catml --selectors random \
    --examples-per-class all --k-shot "$k_shot" --run-test
}

make_fewshot_jobs experiments/configs/clinc150_fewshot_dev.json \
  /home/arseniy/data/clinc150 experiments/jobs/clinc_fewshot_1shot 1
make_fewshot_jobs experiments/configs/clinc150_fewshot_dev.json \
  /home/arseniy/data/clinc150 experiments/jobs/clinc_fewshot_5shot 5
make_fewshot_jobs experiments/configs/cub200_fewshot_dev.json \
  /home/arseniy/data/CUB-200-2011 experiments/jobs/cub_fewshot_1shot 1
make_fewshot_jobs experiments/configs/cub200_fewshot_dev.json \
  /home/arseniy/data/CUB-200-2011 experiments/jobs/cub_fewshot_5shot 5
make_fewshot_jobs experiments/configs/omniglot_fewshot_dev.json \
  /home/arseniy/data/omniglot experiments/jobs/omniglot_fewshot_1shot 1
make_fewshot_jobs experiments/configs/omniglot_fewshot_dev.json \
  /home/arseniy/data/omniglot experiments/jobs/omniglot_fewshot_5shot 5
make_fewshot_jobs experiments/configs/phd08_fewshot_dev.json \
  /home/arseniy/data/PHD08_dataset experiments/jobs/phd08_fewshot_1shot 1
make_fewshot_jobs experiments/configs/phd08_fewshot_dev.json \
  /home/arseniy/data/PHD08_dataset experiments/jobs/phd08_fewshot_5shot 5
```

Все эти job-ы будут обучать модель, выбирать checkpoint на validation и затем
оценивать test. Если пока нужен только development, убери `--run-test` из
функции. После генерации запусти каждую папку `bash <output-dir>/run.sh`.

Для CLINC L5 команда с пятью seed:

```bash
python scripts/run_matrix.py \
  --config experiments/configs/clinc150_open_set.json \
  --data-root /home/arseniy/data/clinc150 \
  --output-dir experiments/jobs/clinc_open_set \
  --seeds 1,2,3,4,5 --losses catml --selectors random \
  --examples-per-class all --run-test
bash experiments/jobs/clinc_open_set/run.sh
```

Validation выбирает порог known/OOS; `--run-test` переносит его на официальный
OOS test. `--stage final` к few-shot/open-set задачам не применим.

### P5, синтетика → PHD08

Сделай отдельную матрицу для каждого алфавита. Для smoke run замени seeds на `1`,
budget на `--epochs 1 --steps-per-epoch 2` и не добавляй `--run-test`. Для
итоговых запусков используй пять seed и `--run-test`:

```bash
python scripts/run_matrix.py \
  --config experiments/configs/phd08_synthetic_2350.json \
  --data-root /home/arseniy/data/kor_synthetic \
  --eval-data-root /home/arseniy/data/PHD08_dataset \
  --output-dir experiments/jobs/p5_2350 \
  --seeds 1,2,3,4,5 --losses cross_entropy --selectors random \
  --examples-per-class all --run-test
bash experiments/jobs/p5_2350/run.sh
```

Для полного алфавита замени конфиг на `phd08_synthetic_full.json` и каталог
вывода на `experiments/jobs/p5_full`. Каждый test job проходит по всем PHD08
изображениям, поэтому сначала запускай один smoke job.

## 5. Что сейчас ещё не покрыто полностью

19 конфигов — это стартовые протоколы, а не автоматическая реализация всех
сравнений из четырёх планов. Через текущие конфиги и матричный генератор
запускаются CATML, triplet, cross-entropy, APM, Auto-clustering и их сочетание,
few-shot, retrieval, CLINC OOS и P5. В планах также есть отдельные ablation-ы,
distance-weighted и другие селекторы, больше loss/baseline систем и точные
исторические протоколы; их нельзя считать выполненными простым запуском всех
19 файлов.

Отдельные ограничения протокола:

- PHD08 P1/P2 сейчас используют случайное разбиение изображений внутри класса;
  без source/font ID нельзя подтвердить запланированную source-disjoint оценку.
- PHD08 P2 с `all,5,10` ограничивает число доступных train-изображений, но не
  обеспечивает отбор по независимым шрифтам/источникам.
- Omniglot O5 запускает локальную `all_runs` папку, но её точное совпадение с
  официальным набором заданий не подтверждено.
- CATML reduction и режим градиента центров задокументированы как вариант,
  пока исторический код не подтверждает точное воспроизведение.

Не включай test-результаты в выбор гиперпараметров. Храни сравниваемые методы
на одинаковых manifest-ах и seed; выводы по нескольким seed считай по всем
запускам, а не по лучшему checkpoint среди test.

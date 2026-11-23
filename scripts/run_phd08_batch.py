"""Prepare and run a resumable PHD08 experiment batch on selected GPUs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from collect_results import DEFAULT_ROOTS, collect


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "experiments" / "configs"
STATE_NAME = "batch_state.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_state(path: Path, state: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".batch-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_matrix(out: Path, metric: str, ce: str, data: Path, models: Path,
                   seeds: str, budgets: str, phase: str, epochs: int | None,
                   steps: int | None, eval_data: Path | None = None) -> list[dict]:
    command = [
        sys.executable, str(ROOT / "scripts" / "run_matrix.py"),
        "--config", str(CONFIGS / f"{metric}.json"),
        "--cross-entropy-config", str(CONFIGS / f"{ce}.json"), "--phd08-baseline",
        "--data-root", str(data), "--models-dir", str(models), "--output-dir", str(out),
        "--seeds", seeds, "--examples-per-class", budgets,
    ]
    # Final class-disjoint P3 retrains on train+validation classes; other tasks
    # retain validation for checkpoint selection and only add test afterwards.
    if phase == "final":
        if metric == "phd08_metric_dev":
            command += ["--stage", "final"]
        command.append("--run-test")
    if eval_data is not None:
        command += ["--eval-data-root", str(eval_data)]
    if epochs is not None:
        command += ["--epochs", str(epochs)]
    if steps is not None:
        command += ["--steps-per-epoch", str(steps)]
    subprocess.run(command, cwd=ROOT, check=True)
    jobs = []
    for line in (out / "jobs.jsonl").read_text(encoding="utf-8").splitlines():
        job = json.loads(line)
        cmd = shlex.split(job["command"])
        cmd[0] = sys.executable
        jobs.append({"id": f"{out.name}__{Path(job['config']).stem}", "command": cmd, "kind": "training", "status": "pending", "attempts": []})
    return jobs


def prepare_p4(p3_dir: Path, phd_root: Path, models: Path) -> list[dict]:
    state_path = p3_dir / STATE_NAME
    if not state_path.is_file():
        raise ValueError(f"P4 needs the completed final P3 batch state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("settings", {}).get("batch") != "p3" or state.get("settings", {}).get("phase") != "final":
        raise ValueError("P4 needs a P3 batch prepared with --phase final")
    if any(job.get("status") != "completed" for job in state["jobs"]):
        raise ValueError("Finish all P3 batch jobs before preparing P4")
    jobs = []
    for job in state["jobs"]:
        checkpoint = Path(job["attempts"][-1]["output"]) / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"P3 checkpoint missing: {checkpoint}")
        for shot in (1, 5):
            config = CONFIGS / f"phd08_fewshot_final_k{shot}.json"
            command = [sys.executable, str(ROOT / "train.py"), "--config", str(config),
                       "--data-root", str(phd_root), "--models-dir", str(models),
                       "--evaluate-pretrained", "--split", "test", "--checkpoint", str(checkpoint),
                       "--output", "OUTPUT_PLACEHOLDER"]
            jobs.append({"id": f"{job['id']}__k{shot}", "command": command, "kind": "evaluation", "status": "pending", "attempts": []})
    return jobs


def prepare(settings: dict, batch_dir: Path) -> dict:
    batch = settings["batch"]
    phd = Path(settings["phd_root"])
    models = Path(settings["models_dir"])
    phase = settings["phase"]
    seeds, epochs, steps = settings["seeds"], settings["epochs"], settings["steps_per_epoch"]
    plan_dir = batch_dir / "plans"
    if batch in ("p1", "p2"):
        jobs = prepare_matrix(plan_dir / batch, "phd08_metric_closed_dev", "phd08_classification",
                              phd, models, seeds, "all" if batch == "p1" else "5,10", phase, epochs, steps)
    elif batch == "p3":
        jobs = prepare_matrix(plan_dir / batch, "phd08_metric_dev", "phd08_metric_ce_dev",
                              phd, models, seeds, "all", phase, epochs, steps)
    elif batch == "p5":
        kor = Path(settings["kor_root"])
        jobs = []
        for size in ("2350", "full"):
            jobs.extend(prepare_matrix(plan_dir / f"p5_{size}", f"phd08_synthetic_metric_{size}",
                                       f"phd08_synthetic_{size}", kor, models, seeds, "all",
                                       phase, epochs, steps, phd))
    else:
        jobs = prepare_p4(Path(settings["p3_dir"]), phd, models)
    return {"schema_version": 1, "created_at": now(), "settings": settings, "jobs": jobs}


def summary(state: dict) -> str:
    counts = Counter(job["status"] for job in state["jobs"])
    return f"total={len(state['jobs'])} " + " ".join(f"{name}={counts.get(name, 0)}" for name in ("pending", "running", "completed", "failed", "interrupted"))


def update_csv(batch_dir: Path) -> None:
    try:
        output = ROOT / "experiments" / "results.csv"
        collect([*DEFAULT_ROOTS, batch_dir], output)
    except Exception as exc:
        print(f"Could not update results.csv: {exc}", file=sys.stderr, flush=True)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def execute(state: dict, state_path: Path, slots: list[str], retry_failed: bool) -> int:
    for job in state["jobs"]:
        if job["status"] == "running":
            attempt = job["attempts"][-1]
            if pid_alive(attempt.get("pid")):
                raise RuntimeError(f"Previous worker PID {attempt['pid']} may still be running; wait for it to stop before restarting")
            attempt.update(status="interrupted", ended_at=now())
            job["status"] = "interrupted"
        if job["status"] == "interrupted" or (retry_failed and job["status"] == "failed"):
            job["status"] = "pending"
    save_state(state_path, state)
    active: dict[subprocess.Popen, tuple[dict, dict, str, object]] = {}
    free = list(slots)
    failed = any(job["status"] == "failed" for job in state["jobs"])
    print(summary(state), flush=True)
    try:
        while True:
            while free and not failed:
                job = next((item for item in state["jobs"] if item["status"] == "pending"), None)
                if job is None:
                    break
                gpu = free.pop(0)
                attempt_no = len(job["attempts"]) + 1
                run_dir = state_path.parent / "runs" / job["id"] / f"attempt_{attempt_no:03d}"
                run_dir.mkdir(parents=True, exist_ok=False)
                output = run_dir / "result.json" if job["kind"] == "evaluation" else run_dir
                log_path = run_dir / "console.log"
                command = list(job["command"])
                command[command.index("--output") + 1] = str(output)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                log = log_path.open("w", encoding="utf-8")
                attempt = {"number": attempt_no, "status": "running", "gpu": gpu, "output": str(output),
                           "log": str(log_path), "started_at": now(), "command": command}
                job["attempts"].append(attempt)
                job["status"] = "running"
                save_state(state_path, state)
                try:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as exc:
                    log.write(f"Could not start worker: {exc}\n")
                    log.close()
                    attempt.update(status="failed", ended_at=now(), error=str(exc))
                    job["status"] = "failed"
                    free.append(gpu)
                    failed = True
                    save_state(state_path, state)
                    break
                attempt["pid"] = proc.pid
                active[proc] = (job, attempt, gpu, log)
                save_state(state_path, state)
                print(f"Started {job['id']} on GPU {gpu}; log: {log_path}", flush=True)
            if not active:
                break
            time.sleep(1)
            for proc, (job, attempt, gpu, log) in list(active.items()):
                returncode = proc.poll()
                if returncode is None:
                    continue
                log.close()
                free.append(gpu)
                attempt.update(status="completed" if returncode == 0 else "failed", ended_at=now(), returncode=returncode)
                job["status"] = attempt["status"]
                failed |= returncode != 0
                del active[proc]
                save_state(state_path, state)
                update_csv(state_path.parent)
                print(f"{job['status']}: {job['id']}; {summary(state)}; log: {attempt['log']}", flush=True)
    except KeyboardInterrupt:
        print("Stopping running workers; completed jobs will be retained.", flush=True)
        for proc in active:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc, (job, attempt, _gpu, log) in active.items():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            log.close()
            attempt.update(status="interrupted", ended_at=now(), returncode=proc.returncode)
            job["status"] = "interrupted"
        save_state(state_path, state)
        update_csv(state_path.parent)
        return 130
    update_csv(state_path.parent)
    print(summary(state), flush=True)
    if failed:
        print("Queue stopped after an error; inspect console.log and restart with --retry-failed.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=("p1", "p2", "p3", "p4", "p5"), required=True)
    parser.add_argument("--phase", choices=("development", "final"), default="development")
    parser.add_argument("--phd-root", type=Path)
    parser.add_argument("--kor-root", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--p3-dir", type=Path, help="Completed final P3 batch directory, required for P4")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Prepare and print the queue without training")
    parser.add_argument("--status", action="store_true", help="Show the saved queue status without running jobs")
    args = parser.parse_args()
    batch_dir = (args.output_dir or ROOT / "experiments" / "jobs" / f"phd08_batch_{args.batch}_{args.phase}").expanduser().resolve()
    state_path = batch_dir / STATE_NAME
    if args.status:
        if not state_path.is_file():
            parser.error(f"Batch state does not exist: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print(summary(state))
        for job in state["jobs"]:
            if job["status"] in ("failed", "interrupted") and job["attempts"]:
                print(f"{job['status']}: {job['id']} — {job['attempts'][-1]['log']}")
        return 0
    if args.phd_root is None or args.models_dir is None:
        parser.error("--phd-root and --models-dir are required to prepare or run a batch")
    if args.batch == "p5" and args.kor_root is None:
        parser.error("--kor-root is required for P5")
    if args.batch == "p4" and (args.phase != "final" or args.p3_dir is None):
        parser.error("P4 requires --phase final and --p3-dir")
    if args.batch == "p4" and (args.epochs is not None or args.steps_per_epoch is not None):
        parser.error("P4 only evaluates existing models; epoch overrides do not apply")
    if args.workers_per_gpu < 1 or (args.epochs is not None and args.epochs < 1) or (args.steps_per_epoch is not None and args.steps_per_epoch < 1):
        parser.error("workers, epochs and steps must be positive")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain distinct GPU IDs")
    seeds = [value.strip() for value in args.seeds.split(",") if value.strip()]
    if not seeds or any(not value.isdigit() for value in seeds) or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain distinct non-negative integers")
    for path in (args.phd_root, args.models_dir, args.kor_root if args.batch == "p5" else None):
        if path is not None and not path.expanduser().is_dir():
            parser.error(f"Directory not found: {path}")
    settings = {
        "batch": args.batch, "phase": args.phase, "phd_root": str(args.phd_root.expanduser().resolve()),
        "kor_root": str(args.kor_root.expanduser().resolve()) if args.batch == "p5" else None,
        "models_dir": str(args.models_dir.expanduser().resolve()),
        "p3_dir": str(args.p3_dir.expanduser().resolve()) if args.batch == "p4" else None,
        "seeds": ",".join(seeds), "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
    }
    batch_dir.mkdir(parents=True, exist_ok=True)
    with (batch_dir / ".batch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another scheduler is already running this batch: {batch_dir}")
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("settings") != settings:
                parser.error("Saved batch has different parameters; use the original parameters or a new --output-dir")
        else:
            state = prepare(settings, batch_dir)
            save_state(state_path, state)
        print(f"Batch: {batch_dir}\n{summary(state)}", flush=True)
        if args.dry_run:
            return 0
        return execute(state, state_path, [gpu for gpu in gpus for _ in range(args.workers_per_gpu)], args.retry_failed)


if __name__ == "__main__":
    raise SystemExit(main())

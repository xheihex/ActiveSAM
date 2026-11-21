"""Small Colab evaluator around the original ActiveSAM model methods."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import ExitStack, contextmanager
import csv
from dataclasses import dataclass, field
import io
import json
from pathlib import Path, PurePosixPath
import runpy
import shutil
import threading
import time
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image


NUM_CLASSES = 150
VALIDATION_IMAGES = 2000
PAPER = {"full": 40.03, "no_epg": 40.05}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_info(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.absolute()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def copy_local(source, target):
    """Copy one Drive file; reuse a copy with matching source metadata and size."""
    source, target = Path(source), Path(target)
    info = source_info(source)
    marker = target.with_name(target.name + ".source.json")
    if target.is_file() and target.stat().st_size == info["bytes"] and read_json(marker) == info:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print("复制到本地：", source.name, flush=True)
    shutil.copyfile(source, target)
    write_json(marker, info)


def validation_members(archive):
    """Select paired validation files, including ZIPs with an ADEChallengeData2016 prefix."""
    members = {}
    for item in archive.infolist():
        parts = PurePosixPath(item.filename).parts
        if (len(parts) < 3 or parts[-3] not in ("images", "annotations")
                or parts[-2] != "validation" or not parts[-1].startswith("ADE_val_")):
            continue
        key = (parts[-3], parts[-1])
        if key in members:
            raise ValueError(f"ZIP 中验证集文件重复：{key}")
        members[key] = item
    expected = {(kind, f"ADE_val_{n:08d}.{suffix}")
                for n in range(1, VALIDATION_IMAGES + 1)
                for kind, suffix in (("images", "jpg"), ("annotations", "png"))}
    if set(members) != expected:
        raise ValueError(f"ZIP 须包含完整验证集图像与原始标注，各 {VALIDATION_IMAGES} 张。")
    return members


def prepare_ade20k(source, root, progress=None):
    """Read one ZIP from Drive, then extract only validation files on local disk."""
    source, root = Path(source), Path(root)
    local_zip = root / "ADE20K.zip"
    copy_local(source, local_zip)
    info = source_info(source)
    marker = root / "dataset.source.json"
    refresh = read_json(marker) != info
    with zipfile.ZipFile(local_zip) as archive:
        members = validation_members(archive)
        iterator = members.items()
        if progress is not None:
            iterator = progress(iterator, total=len(members), desc="准备 ADE20K 验证集", unit="文件")
        for (kind, name), member in iterator:
            target = root / kind / "validation" / name
            if refresh or not target.is_file() or target.stat().st_size != member.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    write_json(marker, info)
    return [(root / "images/validation" / f"ADE_val_{n:08d}.jpg",
             root / "annotations/validation" / f"ADE_val_{n:08d}.png")
            for n in range(1, VALIDATION_IMAGES + 1)]


def load_model_config(repo, checkpoint):
    """Use the two author configuration dictionaries without redefining defaults."""
    repo = Path(repo)
    base = runpy.run_path(str(repo / "configs/base_config.py"))["model"]
    ade = runpy.run_path(str(repo / "configs/cfg_ade20k.py"))["model"]
    canonical = {**base, **ade}
    canonical.pop("type")
    config = {**canonical, "sam3_checkpoint": str(checkpoint)}
    for key in ("vocabulary", "signatures"):
        config[key] = str(repo / canonical[key])
    return config, canonical


def reduce_ade_labels(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.dtype.kind not in "ui" or np.any(raw < 0) or np.any(raw > NUM_CLASSES):
        raise ValueError("须使用 ADE20K 原始单通道标注：0 忽略，1..150 为类别。")
    labels = raw.astype(np.int64) - 1
    labels[raw == 0] = 255
    return labels


def intersect_and_union(prediction, label, num_classes=NUM_CLASSES):
    prediction, label = np.asarray(prediction), np.asarray(label)
    if prediction.shape != label.shape or prediction.ndim != 2:
        raise ValueError("预测与标注须为尺寸相同的二维类别图。")
    if prediction.dtype.kind not in "ui" or np.any(prediction < 0) or np.any(prediction >= num_classes):
        raise ValueError("预测类别 ID 超出范围。")
    valid = label != 255
    pred, truth = prediction[valid].astype(np.int64), label[valid].astype(np.int64)
    # MMSeg IoUMetric uses float32 histograms and sequential float32 summation.
    intersect = np.bincount(truth[pred == truth], minlength=num_classes).astype(np.float32)
    predicted = np.bincount(pred, minlength=num_classes).astype(np.float32)
    labeled = np.bincount(truth, minlength=num_classes).astype(np.float32)
    return np.stack((intersect, predicted + labeled - intersect, predicted, labeled))


def compute_metrics(areas, names):
    intersect, union, predicted, truth = areas
    with np.errstate(divide="ignore", invalid="ignore"):
        iou, acc, overall = intersect / union, intersect / truth, intersect.sum() / truth.sum()
    if not np.isfinite(overall):
        raise ValueError("没有可评测的标注像素。")

    def percent(value):
        return round(float(np.round(np.nanmean(value) * 100, 2)), 2)

    return {
        "mIoU": percent(iou), "mAcc": percent(acc), "aAcc": percent(overall),
        "per_class": [{"class_id": i, "name": name,
                       "IoU": None if not np.isfinite(iou[i]) else percent(iou[i]),
                       "Acc": None if not np.isfinite(acc[i]) else percent(acc[i]),
                       "area_intersect": int(intersect[i]), "area_union": int(union[i]),
                       "area_pred": int(predicted[i]), "area_label": int(truth[i])}
                      for i, name in enumerate(names)],
    }


def check_stop(event):
    if event.is_set():
        raise KeyboardInterrupt("已请求停止；未完成图片将在续跑时重算。")


@contextmanager
def model_interrupt_points(model, event, tau=None):
    """Wrap author method boundaries; their calculations and files stay unchanged."""
    with ExitStack() as stack:
        check_stop(event)
        if tau is not None:
            stack.enter_context(patch.object(model, "tau", tau))
        for name in ("class_maps", "preview", "encode", "run_decoder", "decode_prompts", "fuse_heads", "decode"):
            original = getattr(model, name, None)
            if not callable(original):
                continue

            def guarded(*args, _original=original, **kwargs):
                check_stop(event)
                result = _original(*args, **kwargs)
                check_stop(event)
                return result

            stack.enter_context(patch.object(model, name, guarded))
        yield
        check_stop(event)


@dataclass
class Prediction:
    labels: np.ndarray
    inference_seconds: float
    active_classes: int


class TorchPredictor:
    def __init__(self, model):
        import torch
        self.torch, self.model = torch, model

    def run_interruptible(self, image, event, tau):
        torch = self.torch
        with model_interrupt_points(self.model, event, tau), torch.inference_mode():
            torch.cuda.synchronize()
            started = time.perf_counter()
            maps, presence, active = self.model.class_maps(image, (image.height, image.width))
            labels = self.model.decode(maps, presence, active).detach().cpu().numpy()
            torch.cuda.synchronize()
            return Prediction(labels, time.perf_counter() - started, int(active.numel()))


class InterruptiblePredictor:
    """One model worker; the notebook thread polls and can respond to SIGINT."""
    def __init__(self, predictor, poll_seconds=0.1):
        self.predictor, self.poll_seconds = predictor, poll_seconds
        self.stop_event = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="activesam-model")
        self.future, self.tau, self.closed = None, None, False

    @property
    def busy(self):
        return self.future is not None and not self.future.done()

    def ensure_idle(self):
        if self.busy:
            raise RuntimeError("上一张图仍在结束当前 GPU 调用；请稍后重跑，暂勿重建模型。")

    def set_tau(self, tau):
        self.ensure_idle()
        self.tau = tau

    def request_stop(self):
        self.stop_event.set()
        if self.future is not None:
            self.future.cancel()

    def _dispatch(self, future, image):
        if not future.set_running_or_notify_cancel():
            return
        try:
            check_stop(self.stop_event)
            method = getattr(self.predictor, "run_interruptible", None)
            result = method(image, self.stop_event, self.tau) if callable(method) else self.predictor(image)
            check_stop(self.stop_event)
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(result)
        finally:
            image.close()

    def _release(self, future):
        if not future.cancelled() and self.stop_event.is_set():
            error = future.exception()
            if error is not None:
                error.__traceback__ = None  # Release cancelled frames holding GPU tensors.

    def __call__(self, image):
        if self.closed:
            raise RuntimeError("请重新加载模型。")
        self.ensure_idle()
        self.stop_event.clear()
        owned_image = image.copy()
        # Register before submit: SIGINT inside submit must not lose a running job.
        self.future = Future()
        self.future.add_done_callback(self._release)
        self.future.add_done_callback(lambda future: owned_image.close() if future.cancelled() else None)
        try:
            self.pool.submit(self._dispatch, self.future, owned_image)
            while True:
                check_stop(self.stop_event)
                try:
                    result = self.future.result(timeout=self.poll_seconds)
                except FutureTimeoutError:
                    if not self.future.done():
                        continue
                    result = self.future.result()  # Preserve TimeoutError raised by the model.
                check_stop(self.stop_event)
                return result
        except BaseException:
            self.request_stop()
            raise

    def close(self, *, wait=False):
        self.request_stop()
        if not wait:
            self.ensure_idle()
        self.pool.shutdown(wait=wait, cancel_futures=True)
        self.closed = True


def read_input(paths):
    with Image.open(paths[0]) as source:
        image = source.convert("RGB")
    try:
        with Image.open(paths[1]) as mask:
            label = reduce_ade_labels(np.asarray(mask))
        if label.shape != (image.height, image.width):
            raise ValueError(f"图像与标注尺寸不同：{paths[0].name}")
        return image, label
    except BaseException:
        image.close()
        raise


class OrderedInputLoader:
    """Bounded CPU prefetch in dataset order; input workers never run the model."""
    def __init__(self, entries, workers=2):
        self.entries, self.workers = iter(entries), workers
        self.pending, self.pool = deque(), None

    def __enter__(self):
        if self.workers:
            self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="ade20k-input")
            try:
                self._fill()
            except BaseException:
                self.__exit__(None, None, None)
                raise
        return self

    def _fill(self):
        while len(self.pending) < self.workers * 2:
            paths = next(self.entries, None)
            if paths is None:
                break
            self.pending.append(self.pool.submit(read_input, paths))

    def __next__(self):
        if self.pool is None:
            return read_input(next(self.entries))
        if not self.pending:
            raise StopIteration
        future = self.pending[0]
        while True:
            try:
                prepared = future.result(timeout=.1)
                break
            except FutureTimeoutError:
                if future.done():
                    prepared = future.result()
                    break
        self.pending.popleft()
        try:
            self._fill()
        except BaseException:
            prepared[0].close()
            raise
        return prepared

    def __exit__(self, *_):
        if self.pool is not None:
            def release(future):
                if not future.cancelled() and future.exception() is None:
                    future.result()[0].close()

            for future in self.pending:
                future.cancel()
                future.add_done_callback(release)
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pending.clear()


@dataclass
class EvaluationState:
    areas: np.ndarray = field(default_factory=lambda: np.zeros((4, NUM_CLASSES), dtype=np.float32))
    records: list = field(default_factory=list)


def save_checkpoint(directory, state):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, areas=state.areas, records=np.asarray(json.dumps(state.records)))
    (Path(directory) / "checkpoint.npz").write_bytes(buffer.getvalue())


def load_checkpoint(directory):
    path = Path(directory) / "checkpoint.npz"
    if not path.exists():
        return EvaluationState()
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as data:
        return EvaluationState(data["areas"], json.loads(str(data["records"].item())))


def save_results(directory, summary, state):
    write_csv(directory / "per_class.csv", summary["per_class"])
    write_csv(directory / "per_image.csv", state.records)
    write_json(directory / "results.json", summary)  # Written only after the whole split finishes.


def evaluate(predictor, entries, names, directory, experiment, num_images, *, settings=None,
             checkpoint_every=20, input_workers=2, progress=None):
    if not 1 <= num_images <= len(entries) or checkpoint_every < 1 or input_workers < 0:
        raise ValueError("请检查评测张数、断点间隔和输入线程数。")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = {**(settings or {}), "experiment": experiment, "num_images": num_images}
    previous = read_json(directory / "settings.json")
    if ((directory / "checkpoint.npz").exists() or (directory / "results.json").exists()) and previous != settings:
        raise ValueError("已有结果的模型或输入设置不同；请使用新的结果目录。")
    write_json(directory / "settings.json", settings)
    result = read_json(directory / "results.json")
    if result is not None:
        print("复用已完成结果：", directory, flush=True)
        return result
    entries = entries[:num_images]
    state = load_checkpoint(directory)
    if (state.areas.shape != (4, NUM_CLASSES)
            or [row["id"] for row in state.records] != [paths[0].stem for paths in entries[:len(state.records)]]
            or len(state.records) > num_images):
        raise ValueError("断点与当前验证集不一致。")
    initial_count = len(state.records)
    remaining = entries[initial_count:]
    iterator = remaining if progress is None else progress(
        remaining, total=num_images, initial=initial_count, desc=experiment, unit="张")
    try:
        with OrderedInputLoader(remaining, input_workers) as loader:
            for paths in iterator:
                started = time.perf_counter()
                image, label = next(loader)
                try:
                    prediction = predictor(image)
                    areas = intersect_and_union(prediction.labels, label)
                    record = {"id": paths[0].stem, "width": image.width, "height": image.height,
                              "inference_seconds": prediction.inference_seconds,
                              "pipeline_seconds": time.perf_counter() - started,
                              "active_classes": prediction.active_classes}
                    # One assignment commits the arrays and record together, even across SIGINT.
                    state = EvaluationState(state.areas + areas, state.records + [record])
                    if hasattr(iterator, "set_postfix") and len(state.records) % 10 == 0:
                        iterator.set_postfix(mIoU=compute_metrics(state.areas, names)["mIoU"])
                    if len(state.records) % checkpoint_every == 0:
                        save_checkpoint(directory, state)
                finally:
                    image.close()
        save_checkpoint(directory, state)
    except BaseException:
        if callable(getattr(predictor, "request_stop", None)):
            predictor.request_stop()
        save_checkpoint(directory, state)
        print(f"已停止，已完成 {len(state.records)}/{num_images} 张；等当前 GPU 调用收尾后重跑本格继续。", flush=True)
        raise
    reference = PAPER.get(experiment) if num_images == VALIDATION_IMAGES and len(entries) == VALIDATION_IMAGES else None
    inference_seconds = sum(row["inference_seconds"] for row in state.records)
    pipeline_seconds = sum(row["pipeline_seconds"] for row in state.records)
    result = {"experiment": experiment, "num_images": num_images, **compute_metrics(state.areas, names),
              "paper_mIoU": reference, "resumed_from_images": initial_count,
              "difference_from_paper_pp": None,
              "inference_fps": num_images / inference_seconds,
              "pipeline_fps": num_images / pipeline_seconds,
              "mean_active_classes": float(np.mean([row["active_classes"] for row in state.records]))}
    if reference is not None:
        result["difference_from_paper_pp"] = round(result["mIoU"] - reference, 4)
    save_results(directory, result, state)
    print(f"{experiment}: {num_images} 张，mIoU={result['mIoU']:.2f}", flush=True)
    return result


def write_comparison(directory, results):
    keys = ("experiment", "num_images", "mIoU", "paper_mIoU", "difference_from_paper_pp",
            "inference_fps", "pipeline_fps", "resumed_from_images")
    rows = [{key: result[key] for key in keys} for result in results]
    write_csv(Path(directory) / "comparison.csv", rows)
    write_json(Path(directory) / "comparison.json", rows)
    return rows

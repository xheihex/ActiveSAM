"""CPU checks using in-memory images and I/O; no scratch files or GPU session."""

import _thread
import ast
from contextlib import nullcontext, redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import runpy
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import nbformat
import numpy as np
from PIL import Image

from colab_deployment import reproduce as rep


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RAW = np.array([[0, 1, 2], [150, 1, 2]], dtype=np.uint8)
LABELS = np.array([[149, 0, 1], [149, 0, 1]], dtype=np.uint8)
NAMES = [str(i) for i in range(150)]


def prediction(image):
    return rep.Prediction(LABELS.copy(), .25, 3)


class MetricTests(unittest.TestCase):
    def test_ade_labels_and_ignored_pixels(self):
        label = rep.reduce_ade_labels(RAW)
        np.testing.assert_array_equal(label, [[255, 0, 1], [149, 0, 1]])
        areas = rep.intersect_and_union(LABELS, label)
        self.assertEqual(areas.dtype, np.float32)
        self.assertEqual(areas[3].sum(), 5)
        metrics = rep.compute_metrics(areas, NAMES)
        self.assertEqual((metrics["mIoU"], metrics["mAcc"], metrics["aAcc"]), (100, 100, 100))
        self.assertIsNone(metrics["per_class"][20]["IoU"])

    def test_global_metric_is_not_mean_image_iou(self):
        areas = np.zeros((4, 2), dtype=np.float32)
        for pred, label in (([[0]], [[0]]), (np.ones((1, 99), dtype=int), np.zeros((1, 99), dtype=int))):
            areas += rep.intersect_and_union(np.array(pred), np.array(label), 2)
        result = rep.compute_metrics(areas, ["a", "b"])
        self.assertEqual(result["mIoU"], .5)
        self.assertEqual(result["aAcc"], 1)

    def test_bad_labels_or_predictions_stop(self):
        for raw in (np.zeros((2, 3, 3), dtype=np.uint8), np.array([[151]]), np.array([[-1]])):
            with self.assertRaises(ValueError):
                rep.reduce_ade_labels(raw)
        with self.assertRaises(ValueError):
            rep.intersect_and_union(np.array([[150]]), np.array([[0]]))
        with self.assertRaises(ValueError):
            rep.intersect_and_union(np.zeros((1, 1), dtype=int), np.zeros((1, 2), dtype=int))
        with self.assertRaises(ValueError):
            rep.compute_metrics(np.zeros((4, 150), dtype=np.float32), NAMES)


class DataTests(unittest.TestCase):
    def archive(self, missing=False):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for index in (1, 2):
                for kind, suffix in (("images", "jpg"), ("annotations", "png")):
                    if not (missing and index == 2 and kind == "annotations"):
                        archive.writestr(f"ADEChallengeData2016/{kind}/validation/ADE_val_{index:08d}.{suffix}", b"data")
            archive.writestr("images/training/unneeded.jpg", b"training")
        buffer.seek(0)
        return buffer

    def test_zip_selects_paired_validation_and_ignores_training(self):
        with patch.object(rep, "VALIDATION_IMAGES", 2), zipfile.ZipFile(self.archive()) as archive:
            members = rep.validation_members(archive)
        self.assertEqual(len(members), 4)
        self.assertEqual(set(kind for kind, _ in members), {"images", "annotations"})

    def test_incomplete_zip_is_rejected(self):
        with patch.object(rep, "VALIDATION_IMAGES", 2), zipfile.ZipFile(self.archive(missing=True)) as archive:
            with self.assertRaises(ValueError):
                rep.validation_members(archive)

    def test_local_copy_reuses_metadata_without_reading_source(self):
        info = {"path": "source.zip", "bytes": 100, "mtime_ns": 1}
        with (patch.object(rep, "source_info", return_value=info),
              patch.object(rep, "read_json", return_value=info),
              patch.object(Path, "is_file", return_value=True),
              patch.object(Path, "stat", return_value=SimpleNamespace(st_size=100)),
              patch.object(rep.shutil, "copyfile") as copy):
            rep.copy_local("source.zip", "local.zip")
        copy.assert_not_called()

    def test_extraction_uses_only_validation_paths(self):
        written = {}
        class Sink(io.BytesIO):
            def __init__(self, path):
                super().__init__()
                self.path = path
            def close(self):
                written[str(self.path)] = self.getvalue()
                super().close()
        archive = zipfile.ZipFile(self.archive())
        with (patch.object(rep, "VALIDATION_IMAGES", 2),
              patch.object(rep.zipfile, "ZipFile", return_value=archive),
              patch.object(rep, "copy_local"), patch.object(rep, "source_info", return_value={}),
              patch.object(rep, "read_json"), patch.object(rep, "write_json"),
              patch.object(Path, "is_file", return_value=False), patch.object(Path, "mkdir"),
              patch.object(Path, "open", lambda path, *_a, **_k: Sink(path))):
            entries = rep.prepare_ade20k("source.zip", "ADE20K")
        self.assertEqual(len(written), 4)
        self.assertEqual(len(entries), 2)
        self.assertTrue(all("validation" in path for path in written))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.files, self.state = {}, rep.EvaluationState()
        self.entries = [(Path(f"ADE_val_{i:08d}.jpg"), Path(f"ADE_val_{i:08d}.png")) for i in range(1, 4)]
        state_type = rep.EvaluationState

        def save_checkpoint(_directory, state):
            self.state = state_type(state.areas.copy(), list(state.records))
            self.files["checkpoint.npz"] = True

        self.enterContext(patch.object(Path, "mkdir"))
        self.enterContext(patch.object(Path, "exists", lambda path: path.name in self.files))
        self.enterContext(patch.object(rep, "read_json", lambda path, default=None: self.files.get(path.name, default)))
        self.enterContext(patch.object(rep, "write_json", lambda path, value: self.files.__setitem__(path.name, value)))
        self.enterContext(patch.object(rep, "load_checkpoint", lambda _: state_type(self.state.areas.copy(), list(self.state.records))))
        self.enterContext(patch.object(rep, "save_checkpoint", side_effect=save_checkpoint))
        self.enterContext(patch.object(rep, "save_results", lambda _d, summary, _s: self.files.__setitem__("results.json", summary)))
        self.enterContext(patch.object(rep, "read_input", lambda _: (Image.new("RGB", (3, 2)), rep.reduce_ade_labels(RAW))))
        self.enterContext(redirect_stdout(io.StringIO()))

    def run_eval(self, predictor=prediction, **kwargs):
        return rep.evaluate(predictor, self.entries, NAMES, "memory-results", "no_epg", 3, **kwargs)

    def test_full_and_completed_reuse(self):
        result = self.run_eval(input_workers=2)
        self.assertEqual(result["mIoU"], 100)
        self.assertEqual(result["inference_fps"], 4)
        unused = Mock(side_effect=AssertionError("completed result ran again"))
        self.assertEqual(self.run_eval(unused), result)
        unused.assert_not_called()

    def test_interrupt_saves_prefix_then_resumes_without_double_counting(self):
        first = Mock(side_effect=[prediction(None), KeyboardInterrupt])
        with self.assertRaises(KeyboardInterrupt):
            self.run_eval(first, input_workers=2)
        self.assertEqual(len(self.state.records), 1)
        self.assertNotIn("results.json", self.files)
        resumed = Mock(side_effect=prediction)
        result = self.run_eval(resumed, input_workers=2)
        self.assertEqual(resumed.call_count, 2)
        self.assertEqual(result["resumed_from_images"], 1)
        self.assertEqual(result["mIoU"], 100)
        self.assertEqual(self.state.areas[3].sum(), 15)

    def test_interruption_during_state_commit_does_not_count_half_an_image(self):
        with patch.object(rep, "EvaluationState", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_eval()
        self.assertEqual(len(self.state.records), 0)
        self.assertEqual(self.state.areas.sum(), 0)

    def test_changed_model_settings_cannot_mix_results(self):
        self.run_eval(settings={"tau": 0})
        with self.assertRaises(ValueError):
            self.run_eval(settings={"tau": .1})


class CheckpointTests(unittest.TestCase):
    def test_single_checkpoint_roundtrip_in_memory(self):
        files = {}
        state = rep.EvaluationState(np.ones((4, 150), dtype=np.float32), [{"id": "first"}])
        with (patch.object(Path, "write_bytes", lambda path, value: files.__setitem__(path.name, value)),
              patch.object(Path, "read_bytes", lambda path: files[path.name]),
              patch.object(Path, "exists", lambda path: path.name in files)):
            rep.save_checkpoint(Path("memory-results"), state)
            restored = rep.load_checkpoint("memory-results")
        np.testing.assert_array_equal(restored.areas, state.areas)
        self.assertEqual(restored.records, state.records)
        self.assertEqual(set(files), {"checkpoint.npz"})

    def test_results_export_summary_and_both_csv_tables(self):
        files = {}
        class Sink(io.StringIO):
            def __init__(self, path):
                super().__init__()
                self.name = path.name
            def close(self):
                files[self.name] = self.getvalue()
                super().close()
        state = rep.EvaluationState(rep.intersect_and_union(LABELS, rep.reduce_ade_labels(RAW)), [{"id": "first"}])
        summary = rep.compute_metrics(state.areas, NAMES)
        with (patch.object(Path, "open", lambda path, *_a, **_k: Sink(path)),
              patch.object(Path, "write_text", lambda path, value, **_k: files.__setitem__(path.name, value))):
            rep.save_results(Path("memory-results"), summary, state)
        self.assertEqual(json.loads(files["results.json"])["mIoU"], 100)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(files["per_class.csv"])))), 150)
        self.assertEqual(list(csv.DictReader(io.StringIO(files["per_image.csv"]))), [{"id": "first"}])


class InterruptionTests(unittest.TestCase):
    def test_stop_does_not_wait_for_native_call_and_restores_tau(self):
        entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
        model = SimpleNamespace(tau=.1)
        class Native:
            def run_interruptible(self, image, event, tau):
                with rep.model_interrupt_points(model, event, tau):
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("native call not released")
                    image.getpixel((0, 0))
                    return prediction(image)
        controller = rep.InterruptiblePredictor(Native(), poll_seconds=.01)
        controller.set_tau(0)
        image = Image.new("RGB", (3, 2))
        def caller():
            try:
                controller(image)
            except KeyboardInterrupt:
                stopped.set()
        thread = threading.Thread(target=caller)
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            controller.request_stop()
            self.assertTrue(stopped.wait(1))
            image.close()
            self.assertTrue(controller.busy)
            with self.assertRaises(RuntimeError):
                controller.set_tau(.1)
        finally:
            release.set()
            thread.join(3)
            controller.close(wait=True)
            image.close()
        self.assertEqual(model.tau, .1)
        self.assertIsInstance(controller.future.exception(), KeyboardInterrupt)

    def test_real_sigint_and_interrupt_inside_submit_both_stop_registered_job(self):
        for during_submit in (False, True):
            with self.subTest(during_submit=during_submit):
                entered, release = threading.Event(), threading.Event()
                def native(image):
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("native call not released")
                    image.getpixel((0, 0))
                    return prediction(image)
                controller = rep.InterruptiblePredictor(native, poll_seconds=.01)
                image = Image.new("RGB", (3, 2))
                submit = controller.pool.submit
                def interrupted_submit(*args):
                    submit(*args)
                    self.assertTrue(entered.wait(3))
                    raise KeyboardInterrupt
                def interrupt():
                    if entered.wait(3):
                        _thread.interrupt_main()
                thread = None
                try:
                    if during_submit:
                        context = patch.object(controller.pool, "submit", side_effect=interrupted_submit)
                    else:
                        context = nullcontext()
                        thread = threading.Thread(target=interrupt)
                        thread.start()
                    with context, self.assertRaises(KeyboardInterrupt):
                        controller(image)
                    self.assertTrue(controller.stop_event.is_set())
                    self.assertTrue(controller.busy)
                finally:
                    release.set()
                    if thread:
                        thread.join(3)
                    controller.close(wait=True)
                    image.close()
                self.assertIsInstance(controller.future.exception(), KeyboardInterrupt)

    def test_model_timeout_propagates(self):
        def timeout(_image):
            raise TimeoutError("model timeout")
        controller = rep.InterruptiblePredictor(timeout, poll_seconds=.01)
        with Image.new("RGB", (3, 2)) as image:
            try:
                with self.assertRaisesRegex(TimeoutError, "model timeout"):
                    controller(image)
            finally:
                controller.close(wait=True)

    def test_author_calls_keep_original_dimensions_outputs_and_restore_methods(self):
        output = Mock()
        output.detach.return_value = output
        output.cpu.return_value = output
        output.numpy.return_value = LABELS
        class Model:
            tau = .1
            def class_maps(self, image, shape):
                self.seen = (image.size, shape, self.tau)
                return "maps", "presence", SimpleNamespace(numel=lambda: 3)
            def decode(self, maps, presence, active):
                return output
        model = Model()
        fake_torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(synchronize=lambda: None))
        with patch.dict(sys.modules, {"torch": fake_torch}), Image.new("RGB", (3, 2)) as image:
            result = rep.TorchPredictor(model).run_interruptible(image, threading.Event(), 0)
        self.assertEqual(model.seen, ((3, 2), (2, 3), 0))
        self.assertEqual(model.tau, .1)
        self.assertNotIn("class_maps", vars(model))
        np.testing.assert_array_equal(result.labels, LABELS)


class InputTests(unittest.TestCase):
    def test_ordered_prefetch_and_cleanup(self):
        loaded = []
        def read(entry):
            image = Image.new("RGB", (1, 1), (entry, 0, 0))
            loaded.append(image)
            return image, entry
        loader = rep.OrderedInputLoader(range(1, 9), 2)
        with patch.object(rep, "read_input", side_effect=read):
            with loader:
                for value in range(1, 9):
                    image, label = next(loader)
                    self.assertEqual(label, value)
                    self.assertLessEqual(len(loader.pending), 4)
                    image.close()
            loader.pool.shutdown(wait=True)
        for image in loaded:
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))

    def test_interrupted_exit_does_not_join_blocked_input(self):
        entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
        loaded = []
        def read(entry):
            if entry == 2:
                entered.set()
                if not release.wait(5):
                    raise AssertionError("input call not released")
            image = Image.new("RGB", (1, 1))
            loaded.append(image)
            return image, entry
        loader = rep.OrderedInputLoader([1, 2], 2)
        def caller():
            try:
                with loader:
                    next(loader)[0].close()
                    entered.wait(3)
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                stopped.set()
        with patch.object(rep, "read_input", side_effect=read):
            thread = threading.Thread(target=caller)
            thread.start()
            try:
                self.assertTrue(entered.wait(3))
                self.assertTrue(stopped.wait(1))
            finally:
                release.set()
                thread.join(3)
                loader.pool.shutdown(wait=True)
        for image in loaded:
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))


class NotebookTests(unittest.TestCase):
    def test_clean_notebook_syntax_and_hidden_forms(self):
        document = json.loads((HERE / "ActiveSAM_ADE20K_reproduction_colab.ipynb").read_text(encoding="utf-8"))
        nbformat.validate(nbformat.from_dict(document))
        parameters = {}
        for cell in document["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], "notebook", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["metadata"]["cellView"], "form")
                for line in cell["source"].splitlines():
                    if "# @param" in line:
                        node = ast.parse(line.split("# @param")[0]).body[0]
                        parameters[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(parameters["DATASET_PATH"], "/content/drive/MyDrive/datasets/ADE20K/ADE20K.zip")
        self.assertEqual(parameters["GIT_REF"], "colab-deployment")
        self.assertNotIn("RUN_PIPELINE_BENCHMARK", parameters)

    def test_config_uses_author_values_and_ablation_only_changes_tau(self):
        config, canonical = rep.load_model_config(REPO, "sam3.pt")
        base = runpy.run_path(str(REPO / "configs/base_config.py"))["model"]
        ade = runpy.run_path(str(REPO / "configs/cfg_ade20k.py"))["model"]
        expected = {**base, **ade}
        expected.pop("type")
        self.assertEqual(canonical, expected)
        self.assertEqual(config["vocabulary"], str(REPO / expected["vocabulary"]))
        document = json.loads((HERE / "ActiveSAM_ADE20K_reproduction_colab.ipynb").read_text(encoding="utf-8"))
        functions = [node for cell in document["cells"] if cell["cell_type"] == "code"
                     for node in ast.parse(cell["source"]).body
                     if isinstance(node, ast.FunctionDef) and node.name == "run_experiment"]
        evaluate = Mock(side_effect=lambda *_a, **kwargs: kwargs["settings"]["model"])
        namespace = {"DEFAULT_CONFIG": canonical, "predictor": Mock(), "rep": SimpleNamespace(evaluate=evaluate),
                     "ENTRIES": [], "CLASS_NAMES": NAMES, "RUN_DIRECTORY": Path("memory-results"),
                     "INPUT_SETTINGS": {}, "CHECKPOINT_EVERY": 20, "INFERENCE_PREFETCH_WORKERS": 2, "tqdm": Mock()}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "notebook", "exec"), namespace)
        full = namespace["run_experiment"]("full", 2000)
        no_epg = namespace["run_experiment"]("no_epg", 2000)
        self.assertEqual(full["tau"], .1)
        self.assertEqual(no_epg["tau"], 0)
        self.assertEqual({**full, "tau": 0}, no_epg)

    def test_bootstrap_fetches_selected_branch_without_extra_baseline_fetch(self):
        document = json.loads((HERE / "ActiveSAM_ADE20K_reproduction_colab.ipynb").read_text(encoding="utf-8"))
        functions = [node for cell in document["cells"] if cell["cell_type"] == "code"
                     for node in ast.parse(cell["source"]).body
                     if isinstance(node, ast.FunctionDef) and node.name == "checkout_repository"]
        run = Mock()
        namespace = {"Path": Path, "os": os,
                     "subprocess": SimpleNamespace(run=run, check_output=lambda *_a, **_k: "commit\n")}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "notebook", "exec"), namespace)
        with patch.object(Path, "mkdir"), patch.object(Path, "exists", return_value=False):
            repo, commit = namespace["checkout_repository"]("repository", "colab-deployment", "memory-work")
        self.assertEqual(commit, "commit")
        fetches = [call.args[0] for call in run.call_args_list if "fetch" in call.args[0]]
        self.assertEqual(fetches, [["git", "-C", str(repo), "fetch", "--depth=1", "--", "origin", "colab-deployment"]])


if __name__ == "__main__":
    unittest.main()

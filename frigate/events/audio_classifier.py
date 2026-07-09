"""Utilities for classifying detected audio."""

from __future__ import annotations

import logging
import math
import os
import shutil
import threading
import time
import zipfile
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import requests

from frigate.const import AUDIO_MAX_BIT_RANGE
from frigate.log import suppress_stderr_during
from frigate.util.builtin import load_labels

if TYPE_CHECKING:
    from frigate.config.camera.audio import (
        AudioBirdClassificationConfig,
    )

logger = logging.getLogger(__name__)

BIRDNET_MODEL_FILE_ID = "1ixYBPbZK2Fh1niUQzadE2IWTFZlwATa3"
BIRDNET_DOWNLOAD_URL = (
    f"https://drive.google.com/uc?export=download&id={BIRDNET_MODEL_FILE_ID}"
)
BIRDNET_MODEL_ARCHIVE_NAME = "BirdNET-Analyzer-V2.4.zip"
BIRDNET_MODEL_FILE = "bird_audio_model.tflite"
BIRDNET_LABELMAP_FILE = "bird_audio_labelmap.txt"
BIRDNET_LICENSE_FILE = "bird_audio_model_license.txt"
BIRDNET_LICENSE_TEXT = """BirdNET Analyzer V2.4 pretrained model

Source: https://github.com/birdnet-team/BirdNET-Analyzer/blob/main/docs/models.rst
Download: https://drive.google.com/file/d/1ixYBPbZK2Fh1niUQzadE2IWTFZlwATa3
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International

The BirdNET Analyzer source code is MIT licensed. The published pretrained
models are licensed separately under CC BY-NC-SA 4.0.
"""


class AudioClassification(TypedDict):
    label: str
    score: float


class AudioDetectionPayload(TypedDict, total=False):
    label: str
    score: float
    classifications: list[AudioClassification]


def format_species_label(label: str) -> str:
    """Return the common name from a BirdNET-style label.

    BirdNET labels use "Scientific name_Common Name" where the scientific
    name is a binomial containing a space. Labels that don't match that
    pattern are returned unchanged.
    """
    scientific, _, common = label.partition("_")
    if common and " " in scientific:
        return common
    return label


def resample_audio(
    audio: np.ndarray, source_sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    """Resample a mono audio waveform to the target sample rate."""
    if source_sample_rate == target_sample_rate:
        return audio

    try:
        from scipy import signal

        sample_rate_gcd = math.gcd(source_sample_rate, target_sample_rate)
        up = target_sample_rate // sample_rate_gcd
        down = source_sample_rate // sample_rate_gcd
        return signal.resample_poly(audio.astype(np.float32), up, down).astype(
            np.float32
        )
    except ModuleNotFoundError:
        input_waveform = audio.astype(np.float32)
        sample_count = int(
            round(input_waveform.shape[0] * target_sample_rate / source_sample_rate)
        )
        if sample_count <= 0:
            return np.zeros(0, dtype=np.float32)

        source_positions = np.linspace(0, input_waveform.shape[0] - 1, num=sample_count)
        return np.interp(
            source_positions,
            np.arange(input_waveform.shape[0]),
            input_waveform,
        ).astype(np.float32)


class BirdSoundClassifier:
    """Classify recent audio into bird species using a TFLite waveform model."""

    def __init__(
        self,
        config: AudioBirdClassificationConfig,
        input_sample_rate: int,
        stop_event: threading.Event,
    ) -> None:
        self.config = config
        self.input_sample_rate = input_sample_rate
        self.stop_event = stop_event
        self.enabled = False
        self.last_classification = 0.0
        self.labels: dict[int, str] = {}
        self.buffer = np.zeros(0, dtype=np.float32)
        self.interpreter: Any | None = None
        self.downloader: Any | None = None

        if not self.config.enabled:
            return

        if not self.config.model_path or not self.config.labelmap_path:
            logger.warning(
                "Bird sound classification is enabled but model_path or labelmap_path is not configured"
            )
            return

        if not self._model_files_exist():
            if self._uses_default_model_cache():
                self._download_default_model()
                return

            logger.warning(
                "Bird sound classification model was not found at %s",
                self.config.model_path,
            )
            return

        self._build_classifier()

    def _model_files_exist(self) -> bool:
        return bool(
            self.config.model_path
            and self.config.labelmap_path
            and os.path.exists(self.config.model_path)
            and os.path.exists(self.config.labelmap_path)
        )

    def _uses_default_model_cache(self) -> bool:
        from frigate.config.camera.audio import (
            BIRD_AUDIO_MODEL_CACHE_DIR,
            DEFAULT_BIRD_AUDIO_LABELMAP_PATH,
            DEFAULT_BIRD_AUDIO_MODEL_PATH,
        )

        return (
            self.config.model_path == DEFAULT_BIRD_AUDIO_MODEL_PATH
            and self.config.labelmap_path == DEFAULT_BIRD_AUDIO_LABELMAP_PATH
            and os.path.dirname(self.config.model_path) == BIRD_AUDIO_MODEL_CACHE_DIR
        )

    def _download_default_model(self) -> None:
        from frigate.config.camera.audio import BIRD_AUDIO_MODEL_CACHE_DIR
        from frigate.util.downloader import ModelDownloader

        self.downloader = ModelDownloader(
            model_name="bird_audio",
            download_path=BIRD_AUDIO_MODEL_CACHE_DIR,
            file_names=[
                BIRDNET_MODEL_FILE,
                BIRDNET_LABELMAP_FILE,
                BIRDNET_LICENSE_FILE,
            ],
            download_func=self._download_default_model_file,
            complete_func=self._build_classifier,
        )
        self.downloader.ensure_model_files()

    def _download_default_model_file(self, path: str) -> None:
        file_name = os.path.basename(path)
        if file_name == BIRDNET_LICENSE_FILE:
            with open(path, "w") as f:
                f.write(BIRDNET_LICENSE_TEXT)
            return

        if file_name not in {BIRDNET_MODEL_FILE, BIRDNET_LABELMAP_FILE}:
            raise ValueError(f"Unexpected BirdNET audio model file: {file_name}")

        self._download_and_extract_default_model(os.path.dirname(path))

    def _download_and_extract_default_model(self, download_path: str) -> None:
        os.makedirs(download_path, exist_ok=True)
        archive_path = os.path.join(download_path, BIRDNET_MODEL_ARCHIVE_NAME)

        if not os.path.exists(archive_path):
            logger.info("Downloading BirdNET audio model")
            self._download_from_google_drive(BIRDNET_DOWNLOAD_URL, archive_path)

        with zipfile.ZipFile(archive_path) as archive:
            self._extract_archive_file(
                archive,
                "_Model_FP32.tflite",
                os.path.join(download_path, BIRDNET_MODEL_FILE),
            )
            self._extract_archive_file(
                archive,
                "_Labels.txt",
                os.path.join(download_path, BIRDNET_LABELMAP_FILE),
            )

    def _download_from_google_drive(self, url: str, path: str) -> None:
        with requests.Session() as session:
            response = session.get(url, stream=True, timeout=120)
            response.raise_for_status()

            if "text/html" in response.headers.get("content-type", ""):
                html = response.text
                response.close()
                response = session.get(
                    self._get_google_drive_confirm_url(html),
                    stream=True,
                    timeout=120,
                )
                response.raise_for_status()

            temporary_path = f"{path}.part"
            try:
                with open(temporary_path, "wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            except Exception:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
                raise

            os.replace(temporary_path, path)

    def _get_google_drive_confirm_url(self, html: str) -> str:
        from html.parser import HTMLParser
        from urllib.parse import urlencode

        class DownloadFormParser(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.in_download_form = False
                self.action: str | None = None
                self.params: dict[str, str] = {}

            def handle_starttag(
                self, tag: str, attrs: list[tuple[str, str | None]]
            ) -> None:
                attributes = dict(attrs)
                if tag == "form" and attributes.get("id") == "download-form":
                    self.in_download_form = True
                    self.action = attributes.get("action")
                    return

                if tag == "input" and self.in_download_form:
                    name = attributes.get("name")
                    value = attributes.get("value")
                    if name and value:
                        self.params[name] = value

            def handle_endtag(self, tag: str) -> None:
                if tag == "form":
                    self.in_download_form = False

        parser = DownloadFormParser()
        parser.feed(html)

        if not parser.action or not parser.params:
            raise RuntimeError("Could not find Google Drive download confirmation form")

        return f"{parser.action}?{urlencode(parser.params)}"

    def _extract_archive_file(
        self, archive: zipfile.ZipFile, suffix: str, path: str
    ) -> None:
        if os.path.exists(path):
            return

        matches = [
            name
            for name in archive.namelist()
            if name.endswith(suffix) and "__MACOSX" not in name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not find exactly one {suffix} in BirdNET archive"
            )

        temporary_path = f"{path}.part"
        with archive.open(matches[0]) as source:
            with open(temporary_path, "wb") as output:
                shutil.copyfileobj(source, output)

        os.replace(temporary_path, path)

    def _build_classifier(self) -> None:
        if not self.config.model_path or not self.config.labelmap_path:
            return

        if not os.path.exists(self.config.model_path):
            logger.warning(
                "Bird sound classification model was not found at %s",
                self.config.model_path,
            )
            return

        if not os.path.exists(self.config.labelmap_path):
            logger.warning(
                "Bird sound classification label map was not found at %s",
                self.config.labelmap_path,
            )
            return

        self.labels = load_labels(self.config.labelmap_path, prefill=0)

        try:
            from tflite_runtime.interpreter import Interpreter
        except ModuleNotFoundError:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ModuleNotFoundError:
                logger.warning(
                    "Bird sound classification requires tflite_runtime or ai_edge_litert"
                )
                return

        with suppress_stderr_during("tflite_interpreter_init"):
            self.interpreter = Interpreter(
                model_path=self.config.model_path,
                num_threads=self.config.num_threads,
            )
            self.interpreter.allocate_tensors()

        self.tensor_input_details = self.interpreter.get_input_details()
        self.tensor_output_details = self.interpreter.get_output_details()
        try:
            self.input_shape = self._get_input_shape(self.tensor_input_details[0])
        except ValueError as err:
            logger.warning("%s", err)
            return

        self.window_samples = self._get_window_samples(self.input_shape)
        self.enabled = True

    def _get_input_shape(self, input_details: dict[str, Any]) -> tuple[int, ...]:
        shape = input_details.get("shape_signature", input_details["shape"])
        normalized = [int(dim) for dim in shape]

        if normalized and normalized[0] in (1, -1):
            normalized = normalized[1:]

        if not normalized or any(dim <= 0 for dim in normalized):
            return (int(round(self.config.window_seconds * self.config.sample_rate)),)

        if len(normalized) == 1:
            return (normalized[0],)

        if len(normalized) == 2 and normalized[1] == 1:
            return (normalized[0], 1)

        raise ValueError(
            f"Bird sound classification requires a 1-D waveform input, got {shape}"
        )

    def _get_window_samples(self, input_shape: tuple[int, ...]) -> int:
        return int(input_shape[0])

    def add_audio(self, audio: np.ndarray) -> None:
        """Add a chunk of PCM int16 audio to the rolling classification buffer."""
        if not self.enabled:
            return

        waveform = resample_audio(
            audio, self.input_sample_rate, self.config.sample_rate
        ).astype(np.float32)
        waveform = waveform / AUDIO_MAX_BIT_RANGE

        self.buffer = np.concatenate((self.buffer, waveform))[-self.window_samples :]

    def classify(self) -> list[AudioClassification]:
        """Return accepted bird species classifications for recent audio."""
        if not self.enabled or self.stop_event.is_set():
            return []

        now = time.monotonic()
        if now - self.last_classification < self.config.min_interval:
            return []

        if len(self.buffer) < self.window_samples:
            return []

        self.last_classification = now
        tensor_input = self._prepare_input(self.buffer)
        self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
        self.interpreter.invoke()

        scores = self._read_scores()
        accepted = self._top_scores(scores)

        if accepted:
            logger.debug(
                "Bird sound classifier accepted %s",
                ", ".join(
                    f"{classification['label']}={classification['score']:.2f}"
                    for classification in accepted
                ),
            )

        return accepted

    def _prepare_input(self, audio: np.ndarray) -> np.ndarray:
        input_details = self.tensor_input_details[0]
        dtype = input_details["dtype"]
        tensor_input = audio.astype(np.float32)

        if dtype != np.float32:
            scale, zero_point = input_details.get("quantization", (0, 0))
            if scale:
                tensor_input = np.round((tensor_input / scale) + zero_point)
            tensor_input = np.clip(
                tensor_input,
                np.iinfo(dtype).min,
                np.iinfo(dtype).max,
            ).astype(dtype)

        if len(self.tensor_input_details[0]["shape"]) > len(self.input_shape):
            return np.expand_dims(tensor_input.reshape(self.input_shape), axis=0)

        return tensor_input.reshape(self.input_shape)

    def _read_scores(self) -> np.ndarray:
        output_details = self.tensor_output_details[0]
        output = self.interpreter.get_tensor(output_details["index"])
        scores = np.asarray(output).astype(np.float32).reshape(-1)

        scale, zero_point = output_details.get("quantization", (0, 0))
        if scale:
            scores = (scores - zero_point) * scale

        if self.config.output_activation == "sigmoid":
            scores = 1 / (1 + np.exp(-scores))
        elif self.config.output_activation == "softmax":
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores)
            scores = exp_scores / np.sum(exp_scores)

        return scores

    def _top_scores(self, scores: np.ndarray) -> list[AudioClassification]:
        if scores.size == 0:
            return []

        top_k = min(self.config.top_k, scores.size)
        class_ids = np.argpartition(-scores, top_k - 1)[:top_k]
        class_ids = class_ids[np.argsort(-scores[class_ids])]

        classifications: list[AudioClassification] = []
        for class_id in class_ids:
            score = float(scores[class_id])
            if score < self.config.threshold:
                continue

            label = self.labels.get(int(class_id))
            if not label:
                continue

            classifications.append(
                {"label": format_species_label(label), "score": score}
            )

        return classifications

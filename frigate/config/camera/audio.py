from enum import Enum

from pydantic import Field

from frigate.const import AUDIO_MIN_CONFIDENCE, MODEL_CACHE_DIR

from ..base import FrigateBaseModel

__all__ = [
    "AudioBirdClassificationConfig",
    "AudioClassificationOutputActivationEnum",
    "AudioConfig",
    "AudioFilterConfig",
]


DEFAULT_LISTEN_AUDIO = ["bark", "fire_alarm", "speech", "yell"]
BIRD_AUDIO_MODEL_CACHE_DIR = f"{MODEL_CACHE_DIR}/bird_audio"
DEFAULT_BIRD_AUDIO_MODEL_PATH = f"{BIRD_AUDIO_MODEL_CACHE_DIR}/bird_audio_model.tflite"
DEFAULT_BIRD_AUDIO_LABELMAP_PATH = (
    f"{BIRD_AUDIO_MODEL_CACHE_DIR}/bird_audio_labelmap.txt"
)


class AudioFilterConfig(FrigateBaseModel):
    threshold: float = Field(
        default=0.8,
        ge=AUDIO_MIN_CONFIDENCE,
        lt=1.0,
        title="Minimum audio confidence",
        description="Minimum confidence threshold for the audio event to be counted.",
    )


class AudioClassificationOutputActivationEnum(str, Enum):
    none = "none"
    sigmoid = "sigmoid"
    softmax = "softmax"


class AudioBirdClassificationConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable bird sound classification",
        description="Enable species classification for detected bird audio. Requires a compatible TFLite waveform classifier and label map.",
    )
    model_path: str | None = Field(
        default=DEFAULT_BIRD_AUDIO_MODEL_PATH,
        title="Model path",
        description="Path to a TFLite bird sound classifier model. The default BirdNET model is downloaded to /config/model_cache/bird_audio when bird sound classification is enabled.",
    )
    labelmap_path: str | None = Field(
        default=DEFAULT_BIRD_AUDIO_LABELMAP_PATH,
        title="Label map path",
        description="Path to the classifier label map. The default BirdNET label map is downloaded to /config/model_cache/bird_audio when bird sound classification is enabled.",
    )
    threshold: float = Field(
        default=0.5,
        title="Species threshold",
        description="Minimum score required to accept a bird species classification.",
        gt=0.0,
        le=1.0,
    )
    trigger_labels: list[str] = Field(
        default=["bird"],
        title="Trigger labels",
        description="Audio labels that trigger species classification when detected.",
    )
    sample_rate: int = Field(
        default=48000,
        title="Classifier sample rate",
        description="Audio sample rate expected by the bird sound classifier.",
        ge=8000,
        le=96000,
    )
    window_seconds: float = Field(
        default=3.0,
        title="Classification window",
        description="Number of seconds of recent audio to use when the classifier has a dynamic input shape.",
        gt=0.0,
        le=30.0,
    )
    min_interval: float = Field(
        default=10.0,
        title="Minimum interval",
        description="Minimum seconds between bird sound classification attempts per camera.",
        gt=0.0,
    )
    top_k: int = Field(
        default=3,
        title="Top species",
        description="Maximum number of accepted species candidates to expose in audio detection metadata.",
        ge=1,
        le=10,
    )
    num_threads: int = Field(
        default=2,
        title="Classification threads",
        description="Number of threads to use for bird sound classification.",
        ge=1,
    )
    output_activation: AudioClassificationOutputActivationEnum = Field(
        default=AudioClassificationOutputActivationEnum.sigmoid,
        title="Output activation",
        description="Activation to apply to raw classifier output before thresholding. The default BirdNET model outputs raw logits, so sigmoid is applied by default. Set to none for models that already output probabilities.",
    )


class AudioConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable audio detection",
        description="Enable or disable audio event detection for all cameras; can be overridden per-camera.",
    )
    max_not_heard: int = Field(
        default=30,
        title="End timeout",
        description="Amount of seconds without the configured audio type before the audio event is ended.",
    )
    min_volume: int = Field(
        default=500,
        title="Minimum volume",
        description="Minimum RMS volume threshold required to run audio detection; lower values increase sensitivity (e.g., 200 high, 500 medium, 1000 low).",
    )
    listen: list[str] = Field(
        default=DEFAULT_LISTEN_AUDIO,
        title="Listen types",
        description="List of audio event types to detect (for example: bark, fire_alarm, speech, yell).",
    )
    filters: dict[str, AudioFilterConfig] | None = Field(
        None,
        title="Audio filters",
        description="Per-audio-type filter settings such as confidence thresholds used to reduce false positives.",
    )
    enabled_in_config: bool | None = Field(
        None,
        title="Original audio state",
        description="Indicates whether audio detection was originally enabled in the static config file.",
    )
    num_threads: int = Field(
        default=2,
        title="Detection threads",
        description="Number of threads to use for audio detection processing.",
        ge=1,
    )
    bird_classification: AudioBirdClassificationConfig = Field(
        default_factory=AudioBirdClassificationConfig,
        title="Bird sound classification",
        description="Settings for classifying detected bird audio into species.",
    )

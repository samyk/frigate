import sys
import types
import unittest
from unittest.mock import Mock

# stub heavy optional runtime deps so the review maintainer imports in
# environments without them; setdefault keeps the real modules when present
sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("librosa", Mock())
sys.modules.setdefault("soundfile", Mock())
if "sherpa_onnx" not in sys.modules:
    sherpa_stub = types.ModuleType("sherpa_onnx")
    sherpa_stub.OnlineRecognizer = type("OnlineRecognizer", (), {})
    sys.modules["sherpa_onnx"] = sherpa_stub

from frigate.events.audio_classifier import format_species_label  # noqa: E402
from frigate.review.maintainer import get_audio_detection_info  # noqa: E402


class TestFormatSpeciesLabel(unittest.TestCase):
    def test_birdnet_label_returns_common_name(self):
        assert (
            format_species_label("Mimus polyglottos_Northern Mockingbird")
            == "Northern Mockingbird"
        )

    def test_label_with_underscores_but_no_scientific_name_is_unchanged(self):
        assert format_species_label("great_horned_owl") == "great_horned_owl"

    def test_label_without_separator_is_unchanged(self):
        assert format_species_label("Engine") == "Engine"


class TestGetAudioDetectionInfo(unittest.TestCase):
    def test_plain_label_has_no_species(self):
        assert get_audio_detection_info("bird") == ("bird", None)

    def test_detection_without_classifications_has_no_species(self):
        detection = {"label": "bird", "score": 0.9}
        assert get_audio_detection_info(detection) == ("bird", None)

    def test_detection_with_classifications_returns_best_species(self):
        detection = {
            "label": "bird",
            "score": 0.9,
            "classifications": [
                {"label": "House Sparrow", "score": 0.61},
                {"label": "Mexican Jay", "score": 0.83},
            ],
        }
        assert get_audio_detection_info(detection) == ("bird", "Mexican Jay")


if __name__ == "__main__":
    unittest.main()

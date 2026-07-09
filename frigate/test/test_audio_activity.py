import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.modules.setdefault("cv2", Mock())

from frigate.camera.activity_manager import AudioActivityManager  # noqa: E402
from frigate.comms.event_metadata_updater import EventMetadataTypeEnum  # noqa: E402


def create_manager():
    audio_config = SimpleNamespace(enabled_in_config=True, max_not_heard=30)
    camera_config = SimpleNamespace(name="front", audio=audio_config)
    config = SimpleNamespace(cameras={"front": camera_config})
    published = []
    publisher = Mock()

    with patch(
        "frigate.camera.activity_manager.EventMetadataPublisher",
        return_value=publisher,
    ):
        manager = AudioActivityManager(
            config,
            lambda topic, payload: published.append((topic, payload)),
        )

    return manager, published, publisher


class TestAudioActivityManager(unittest.TestCase):
    def test_structured_audio_detection_sets_species_sub_label(self):
        manager, published, publisher = create_manager()

        manager.update_activity(
            {
                "front": {
                    "detections": [
                        {
                            "label": "bird",
                            "score": 0.91,
                            "classifications": [
                                {"label": "Black-capped Chickadee", "score": 0.84}
                            ],
                        }
                    ]
                }
            }
        )

        metadata_payload, metadata_topic = publisher.publish.call_args.args
        assert metadata_topic == EventMetadataTypeEnum.manual_event_create.value
        assert metadata_payload[2] == "bird"
        assert metadata_payload[5] == 0.91
        assert metadata_payload[6] == "Black-capped Chickadee"

        audio_payload = [
            payload for topic, payload in published if topic == "audio_detections"
        ][0]
        audio_detections = json.loads(audio_payload)
        classification = audio_detections["front"]["bird"]["classifications"][0]

        assert classification["label"] == "Black-capped Chickadee"
        assert classification["score"] == 0.84

    def test_existing_audio_detection_updates_species_sub_label(self):
        manager, _, publisher = create_manager()

        manager.compare_audio_activity("front", [("bird", 0.8)], 1.0)
        manager.compare_audio_activity(
            "front",
            [
                {
                    "label": "bird",
                    "score": 0.9,
                    "classifications": [{"label": "Northern Cardinal", "score": 0.77}],
                }
            ],
            2.0,
        )

        metadata_payload, metadata_topic = publisher.publish.call_args.args
        assert metadata_topic == EventMetadataTypeEnum.sub_label.value
        assert metadata_payload[1] == "Northern Cardinal"
        assert metadata_payload[2] == 0.77


if __name__ == "__main__":
    unittest.main()

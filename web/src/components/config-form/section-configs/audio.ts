import type { SectionConfigOverrides } from "./types";

const audio: SectionConfigOverrides = {
  base: {
    sectionDocs: "/configuration/audio_detectors",
    messages: [
      {
        key: "no-audio-role",
        messageKey: "configMessages.audio.noAudioRole",
        severity: "warning",
        condition: (ctx) => {
          if (ctx.level === "camera" && ctx.fullCameraConfig) {
            return !ctx.fullCameraConfig.ffmpeg?.inputs?.some((input) =>
              input.roles?.includes("audio"),
            );
          }
          return false;
        },
      },
    ],
    restartRequired: [],
    fieldOrder: [
      "enabled",
      "listen",
      "filters",
      "min_volume",
      "max_not_heard",
      "num_threads",
      "bird_classification",
      "bird_classification.enabled",
      "bird_classification.threshold",
      "bird_classification.trigger_labels",
      "bird_classification.model_path",
      "bird_classification.labelmap_path",
      "bird_classification.sample_rate",
      "bird_classification.window_seconds",
      "bird_classification.min_interval",
      "bird_classification.top_k",
      "bird_classification.num_threads",
      "bird_classification.output_activation",
    ],
    fieldGroups: {
      detection: ["listen", "filters"],
      sensitivity: ["min_volume", "max_not_heard"],
      classification: [
        "bird_classification",
        "bird_classification.enabled",
        "bird_classification.threshold",
        "bird_classification.trigger_labels",
        "bird_classification.model_path",
        "bird_classification.labelmap_path",
        "bird_classification.sample_rate",
        "bird_classification.window_seconds",
        "bird_classification.min_interval",
        "bird_classification.top_k",
        "bird_classification.num_threads",
        "bird_classification.output_activation",
      ],
    },
    hiddenFields: ["enabled_in_config"],
    advancedFields: [
      "min_volume",
      "max_not_heard",
      "num_threads",
      "bird_classification.model_path",
      "bird_classification.labelmap_path",
      "bird_classification.sample_rate",
      "bird_classification.window_seconds",
      "bird_classification.min_interval",
      "bird_classification.top_k",
      "bird_classification.num_threads",
      "bird_classification.output_activation",
    ],
    uiSchema: {
      filters: {
        "ui:options": {
          expandable: false,
        },
      },
      "filters.*": {
        "ui:options": {
          additionalPropertyKeyReadonly: true,
          isAudioLabels: true,
        },
      },
      listen: {
        "ui:widget": "audioLabels",
      },
    },
  },
  global: {
    restartRequired: [
      "num_threads",
      "bird_classification.enabled",
      "bird_classification.model_path",
      "bird_classification.labelmap_path",
      "bird_classification.sample_rate",
      "bird_classification.num_threads",
    ],
  },
  camera: {
    restartRequired: [
      "num_threads",
      "bird_classification.enabled",
      "bird_classification.model_path",
      "bird_classification.labelmap_path",
      "bird_classification.sample_rate",
      "bird_classification.num_threads",
    ],
  },
};

export default audio;

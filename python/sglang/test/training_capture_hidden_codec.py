"""Compatibility imports for experimental hidden readback scripts."""

from sglang.srt.training_capture.codec import (
    HIDDEN_CODEC,
    HIDDEN_SIZE,
    HiddenManifest,
)
from sglang.srt.training_capture.codec import (
    HIDDEN_CONTRACT as CONTRACT,
)
from sglang.srt.training_capture.protocol import Manifest as NORMAL_MANIFEST

aux_specs = HIDDEN_CODEC.aux_specs
decode_manifest = HIDDEN_CODEC.decode_manifest
validate_manifest = HIDDEN_CODEC.validate_manifest
validate_tensors = HIDDEN_CODEC.validate_tensors

__all__ = [
    "CONTRACT",
    "HIDDEN_SIZE",
    "HiddenManifest",
    "NORMAL_MANIFEST",
    "aux_specs",
    "decode_manifest",
    "validate_manifest",
    "validate_tensors",
]

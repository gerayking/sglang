"""Explicit teacher layouts and codecs; selection never changes module globals."""

from dataclasses import dataclass
from typing import Literal

from sglang.srt.training_capture import protocol as p

TOP128_CONTRACT = "maas-target-kv-top128-v1"
HIDDEN_CONTRACT = "experimental-target-kv-final-hidden-v1"
HIDDEN_SIZE = 2560


class HiddenManifest(p.Manifest):
    contract_id: Literal[HIDDEN_CONTRACT] = HIDDEN_CONTRACT
    payload_format: Literal["experimental_target_kv_final_hidden_v1"] = (
        "experimental_target_kv_final_hidden_v1"
    )
    logits: None = None
    hidden_size: Literal[2560] = HIDDEN_SIZE
    hidden_semantics: Literal[
        "post_final_norm_lm_head_input_predicts_logits_position"
    ] = "post_final_norm_lm_head_input_predicts_logits_position"


@dataclass(frozen=True)
class TeacherSpec:
    kind: Literal["top128", "final_hidden"]
    hidden_size: int = 0

    @property
    def tensor_names(self):
        if self.kind == "final_hidden":
            return ("teacher_final_hidden",)
        return ("teacher_topk_ids", "teacher_topk_logits", "teacher_logsumexp")

    @property
    def output_field(self):
        return (
            "training_capture_hidden_states"
            if self.kind == "final_hidden"
            else "next_token_logits"
        )


@dataclass(frozen=True)
class SnapshotCodec:
    contract_id: str
    payload_format: str
    manifest_type: type[p.Manifest]
    teacher: TeacherSpec

    def aux_specs(self, n, r):
        specs = p.aux_specs(n, r)
        if self.teacher.kind == "final_hidden":
            for name in TOP128_CODEC.teacher.tensor_names:
                del specs[name]
            specs["teacher_final_hidden"] = ("bfloat16", [r, self.teacher.hidden_size])
        return specs

    def validate_manifest(self, manifest, **kwargs):
        if manifest.contract_id != self.contract_id:
            raise p.ContractError("manifest differs from the selected capture codec")
        return p.validate_manifest(manifest, _codec=self, **kwargs)

    def validate_tensors(self, manifest, tensors, **kwargs):
        return p.validate_tensors(manifest, tensors, _codec=self, **kwargs)

    def decode_manifest(self, data, **kwargs):
        return p.decode_manifest(data, _codec=self, **kwargs)


TOP128_CODEC = SnapshotCodec(
    TOP128_CONTRACT, "maas_target_kv_v1", p.Manifest, TeacherSpec("top128")
)
HIDDEN_CODEC = SnapshotCodec(
    HIDDEN_CONTRACT,
    "experimental_target_kv_final_hidden_v1",
    HiddenManifest,
    TeacherSpec("final_hidden", HIDDEN_SIZE),
)


def get_codec(contract_id=TOP128_CONTRACT):
    if contract_id == TOP128_CONTRACT:
        return TOP128_CODEC
    if contract_id == HIDDEN_CONTRACT:
        return HIDDEN_CODEC
    raise p.ContractError(f"unsupported capture contract: {contract_id}")

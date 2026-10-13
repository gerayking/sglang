"""Experimental Qwen3.5-4B hidden payload. Never enable in production readers.

This deliberately uses a different wire format: it contains no teacher logits.
The normal top-128 decoder must reject it. Only the experiment installs this codec.
"""

import math
import sys
from collections import defaultdict
from datetime import datetime
from typing import Literal

import msgspec
import torch

from sglang.srt.training_capture import protocol as p

HIDDEN_SIZE = 2560
CONTRACT = "experimental-target-kv-final-hidden-v1"
NORMAL_AUX = p.aux_specs
NORMAL_MANIFEST = p.Manifest
INSTALLED = False


class HiddenManifest(NORMAL_MANIFEST):
    contract_id: Literal[CONTRACT] = CONTRACT
    payload_format: Literal["experimental_target_kv_final_hidden_v1"] = (
        "experimental_target_kv_final_hidden_v1"
    )
    logits: None = None
    hidden_size: Literal[2560] = HIDDEN_SIZE
    hidden_semantics: Literal[
        "post_final_norm_lm_head_input_predicts_logits_position"
    ] = "post_final_norm_lm_head_input_predicts_logits_position"


def aux_specs(n, r):
    specs = NORMAL_AUX(n, r)
    for name in ("teacher_topk_ids", "teacher_topk_logits", "teacher_logsumexp"):
        del specs[name]
    specs["teacher_final_hidden"] = ("bfloat16", [r, HIDDEN_SIZE])
    return specs


def validate_manifest(manifest, *, max_tensor_bytes=2 << 30):
    try:
        msgspec.json.decode(msgspec.json.encode(manifest), type=HiddenManifest)
        created = datetime.fromisoformat(manifest.created_at.replace("Z", "+00:00"))
        assert created.tzinfo is not None
    except (msgspec.ValidationError, ValueError, TypeError, AssertionError) as error:
        raise p.ContractError("invalid experimental hidden manifest") from error
    n, r = manifest.sequence.total_length, manifest.sequence.response_length
    if n != manifest.sequence.prompt_length + r or manifest.extensions.get(
        "example_only"
    ):
        raise p.ContractError("invalid sequence or example payload")
    if (
        len(manifest.objects) > 16384
        or not 0 < manifest.total_tensor_bytes <= max_tensor_bytes
    ):
        raise p.ContractError("hidden sample exceeds allocation budget")
    p.validate_kv_spec(manifest.kv)
    owners = manifest.topology.owners
    if len(set(owners)) != len(owners) or manifest.topology.aux_owner not in owners:
        raise p.ContractError("invalid owners")
    specs = aux_specs(n, r)
    ids, keys, aux = set(), set(), set()
    groups = defaultdict(list)
    geometry = {layer.layer_id: layer for layer in manifest.kv.layers}
    total = 0
    for obj in manifest.objects:
        if obj.object_id in ids or obj.key in keys or obj.owner_id not in owners:
            raise p.ContractError("duplicate object or invalid owner")
        ids.add(obj.object_id)
        keys.add(obj.key)
        if (
            not obj.contiguous
            or obj.nbytes != math.prod(obj.shape) * p.ELEMENT_BYTES[obj.dtype]
        ):
            raise p.ContractError("invalid tensor layout")
        if not obj.key.startswith(manifest.key_prefix) or ".." in obj.key.split("/"):
            raise p.ContractError("invalid object namespace")
        total += obj.nbytes
        if total > max_tensor_bytes:
            raise p.ContractError("tensor allocation budget exceeded")
        if obj.kind == "aux":
            if obj.name in aux or (obj.dtype, obj.shape) != specs.get(obj.name):
                raise p.ContractError("invalid auxiliary descriptor")
            if (
                obj.key != manifest.key_prefix + "aux/" + obj.name
                or obj.owner_id != manifest.topology.aux_owner
            ):
                raise p.ContractError("invalid auxiliary key/owner")
            if any(
                value is not None
                for value in (
                    obj.layer_id,
                    obj.component,
                    obj.token_range,
                    obj.head_range,
                )
            ):
                raise p.ContractError("auxiliary object has KV ranges")
            aux.add(obj.name)
        else:
            if (
                obj.layer_id not in geometry
                or obj.component not in ("k", "v")
                or obj.token_range is None
                or obj.head_range is None
            ):
                raise p.ContractError("invalid KV descriptor")
            layer = geometry[obj.layer_id]
            t0, t1 = obj.token_range
            h0, h1 = obj.head_range
            dim = layer.key_head_dim if obj.component == "k" else layer.value_head_dim
            if not (0 <= t0 < t1 <= n and 0 <= h0 < h1 <= layer.num_kv_heads):
                raise p.ContractError("invalid KV range")
            if (
                obj.shape != [t1 - t0, h1 - h0, dim]
                or obj.dtype != manifest.kv.dtype
                or obj.name != f"target_{obj.component}.{obj.layer_id}"
            ):
                raise p.ContractError("invalid KV geometry")
            stem = manifest.key_prefix + f"kv/{obj.layer_id}/{obj.owner_id}/"
            suffix = obj.key.removeprefix(stem).split("/")
            if (
                not obj.key.startswith(stem)
                or len(suffix) != 2
                or not suffix[0].isdigit()
                or suffix[1] != obj.component
            ):
                raise p.ContractError("invalid KV key")
            groups[obj.layer_id, obj.component].append(obj)
    if aux != set(specs) or total != manifest.total_tensor_bytes:
        raise p.ContractError("missing auxiliary payload or inconsistent bytes")
    if {obj.owner_id for obj in manifest.objects} != set(owners):
        raise p.ContractError("missing owner objects")
    coverage = {
        p._coverage(groups[layer.layer_id, component], n, layer.num_kv_heads)
        for layer in manifest.kv.layers
        for component in ("k", "v")
    }
    if len(coverage) != 1:
        raise p.ContractError("inconsistent KV coverage")
    return coverage.pop()


def decode_manifest(data, *, max_manifest_bytes=8 << 20, max_tensor_bytes=2 << 30):
    if len(data) > max_manifest_bytes:
        raise p.ContractError("manifest exceeds metadata budget")
    manifest = msgspec.json.decode(data, type=HiddenManifest)
    raw = msgspec.json.decode(data)
    if not (set(HiddenManifest.__struct_fields__) - {"extensions"}).issubset(raw):
        raise p.ContractError("missing manifest fields")
    validate_manifest(manifest, max_tensor_bytes=max_tensor_bytes)
    return manifest


def validate_tensors(
    manifest, tensors, *, max_tensor_bytes=2 << 30, owner_id=None, payload_hasher=None
):
    nv = validate_manifest(manifest, max_tensor_bytes=max_tensor_bytes)
    if owner_id is not None and owner_id not in manifest.topology.owners:
        raise p.ContractError("unregistered tensor owner")
    objects = [
        obj for obj in manifest.objects if owner_id is None or obj.owner_id == owner_id
    ]
    if set(tensors) != {obj.key for obj in objects}:
        raise p.ContractError("tensor object set differs")
    payloads, aux = [], {}
    for obj in objects:
        value = tensors[obj.key]
        if list(value.shape) != obj.shape or value.dtype != p.DTYPES[obj.dtype]:
            raise p.ContractError("tensor shape/dtype mismatch")
        payloads.append(p.tensor_bytes(value))
    digests = (
        payload_hasher.digests(payloads)
        if payload_hasher
        else [p.digest_bytes(data) for data in payloads]
    )
    for obj, data, digest in zip(objects, payloads, digests, strict=True):
        value = tensors[obj.key]
        if digest != obj.sha256:
            raise p.ContractError("tensor checksum mismatch")
        if value.is_floating_point() and not p._all_finite(data, obj.dtype):
            raise p.ContractError("nonfinite captured tensor")
        if obj.kind == "aux":
            aux[obj.name] = value
    if owner_id is not None and owner_id != manifest.topology.aux_owner:
        return
    n, prompt = manifest.sequence.total_length, manifest.sequence.prompt_length
    if not (
        (aux["token_ids"] >= 0) & (aux["token_ids"] < manifest.teacher.vocab_size)
    ).all():
        raise p.ContractError("invalid token IDs")
    if not torch.equal(
        aux["logits_positions"], torch.arange(prompt, n, dtype=torch.int32)
    ):
        raise p.ContractError("hidden rows do not cover accepted response tokens")
    positions = aux["position_ids"]
    if not ((positions >= 0).all() and (positions[1:] > positions[:-1]).all()):
        raise p.ContractError("invalid positions")
    mask = aux["loss_mask"]
    if (
        mask[:prompt].any()
        or (mask > 1).any()
        or (
            manifest.sequence.loss_mask_policy == "current_response_include_eos"
            and not mask[prompt:].all()
        )
    ):
        raise p.ContractError("invalid response mask")
    expected = torch.zeros(n, dtype=torch.uint8)
    expected[:nv] = 1
    if not torch.equal(aux["kv_valid"], expected):
        raise p.ContractError("invalid KV coverage mask")


def install():
    global INSTALLED
    if INSTALLED:
        return
    replacements = {
        "Manifest": HiddenManifest,
        "aux_specs": aux_specs,
        "decode_manifest": decode_manifest,
        "validate_manifest": validate_manifest,
        "validate_tensors": validate_tensors,
    }
    originals = {name: getattr(p, name) for name in replacements}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if not name.startswith("sglang.srt.training_capture"):
            continue
        for key, value in replacements.items():
            if getattr(module, key, None) is originals[key]:
                setattr(module, key, value)
    from sglang.srt.training_capture.catalog import HTTPCaptureCatalog

    def check_hidden_compatibility(self, *, contract_id, kv_codec, store_protocol):
        caps = self.capabilities()
        if not (
            caps.protocol_version == 1
            and caps.hard_pin
            and caps.retention_policy == "retain_until_checkpoint"
            and caps.max_request_bytes >= 8 << 20
            and store_protocol in caps.store_protocols
            and any(
                item.contract_id == contract_id == CONTRACT
                and item.schema_version == 1
                and item.payload_format == "experimental_target_kv_final_hidden_v1"
                and kv_codec in item.kv_codecs
                for item in caps.contracts
            )
        ):
            raise p.ContractError("Catalog does not support experimental hidden data")
        return caps

    HTTPCaptureCatalog.check_compatibility = check_hidden_compatibility
    INSTALLED = True


def writer_factory(config):
    from sglang.srt.training_capture.writer_process import _connect_writer

    install()
    return _connect_writer(config)

"""Out-of-band integrity and LM-head alignment checks for hidden probe archives."""

import argparse
import json
from pathlib import Path

import msgspec
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from sglang.srt.training_capture.protocol import ContractError
from sglang.test.training_capture_hidden_codec import (
    NORMAL_MANIFEST,
    decode_manifest,
    validate_tensors,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--archives", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((args.model / "config.json").read_text())
    assert (
        config["tie_word_embeddings"] and config["text_config"]["hidden_size"] == 2560
    )
    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    key = next(key for key in weight_map if key.endswith("embed_tokens.weight"))
    with safe_open(
        args.model / weight_map[key], framework="pt", device="cpu"
    ) as source:
        weight = source.get_tensor(key).cuda()
    result = {
        "samples": 0,
        "rows": 0,
        "argmax_matches": 0,
        "accepted_tokens_at_max": 0,
        "legacy_decoder_rejections": 0,
        "corruption_rejections": 0,
        "duplicate_archived_samples": 0,
        "max_logit_gap": 0.0,
        "weight_key": key,
        "scope": "LM head only; no transformer prefill. Untimed validation. BF16 GEMM batch size can change near-tie rounding.",
    }
    seen = set()
    for root in args.archives:
        for path in sorted(root.glob("sample-*/manifest.json")):
            data = path.read_bytes()
            manifest = decode_manifest(data)
            identity = (manifest.dataset_id, manifest.sample_id, manifest.generation_id)
            if identity in seen:
                result["duplicate_archived_samples"] += 1
                continue
            seen.add(identity)
            tensors = load_file(path.with_name("tensors.safetensors"))
            validate_tensors(manifest, tensors)
            try:
                msgspec.json.decode(data, type=NORMAL_MANIFEST)
            except msgspec.ValidationError:
                result["legacy_decoder_rejections"] += 1
            else:
                raise RuntimeError("legacy decoder unexpectedly accepted hidden data")
            aux = {
                obj.name: tensors[obj.key]
                for obj in manifest.objects
                if obj.kind == "aux"
            }
            hidden = aux["teacher_final_hidden"]
            # A byte-corruption check exercises the actual archive's payload hash.
            hidden.view(torch.uint8).flatten()[0] ^= 1
            try:
                validate_tensors(manifest, tensors)
            except ContractError:
                result["corruption_rejections"] += 1
            else:
                raise RuntimeError("corrupt hidden passed validation")
            finally:
                hidden.view(torch.uint8).flatten()[0] ^= 1
            logits = F.linear(hidden.cuda(), weight).float()
            expected = aux["token_ids"][manifest.sequence.prompt_length :].long().cuda()
            best, predicted = logits.max(-1)
            accepted = logits.gather(1, expected[:, None]).squeeze(1)
            result["samples"] += 1
            result["rows"] += expected.numel()
            result["argmax_matches"] += int((predicted == expected).sum())
            result["accepted_tokens_at_max"] += int((accepted == best).sum())
            result["max_logit_gap"] = max(
                result["max_logit_gap"], float((best - accepted).max())
            )
    if not result["rows"]:
        raise RuntimeError("no archived hidden samples")
    result["argmax_agreement"] = result["argmax_matches"] / result["rows"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["accepted_tokens_at_max"] != result["rows"]:
        raise RuntimeError(
            "reconstructed greedy logits disagree; inspect numerical gaps"
        )


if __name__ == "__main__":
    main()

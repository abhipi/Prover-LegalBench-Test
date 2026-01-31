import json
import random
from typing import Any, Dict, List, Tuple, Optional

from task_configs import TaskConfig


def _example_to_kv(example: Dict[str, Any], label_key: str) -> Dict[str, Any]:
    """Return example fields excluding label."""
    return {k: v for k, v in example.items() if k != label_key}


def build_prompt(
    task: TaskConfig,
    test_ex: Dict[str, Any],
    fewshot: List[Tuple[Dict[str, Any], Any]],
    label_key: str,
) -> str:
    """Generic prompt builder that works across LegalBench tasks.

    It prints a compact JSON for each example's inputs to avoid guessing field names.
    """
    parts: List[str] = []
    parts.append(task.instruction)
    parts.append("")
    if fewshot:
        parts.append("Examples:")
        for i, (ex, y) in enumerate(fewshot, 1):
            x = _example_to_kv(ex, label_key)
            parts.append(f"[Example {i} Input]")
            parts.append(json.dumps(x, ensure_ascii=False, indent=2))
            parts.append(f"[Example {i} Answer]")
            parts.append(str(y))
            parts.append("")
    parts.append("Now answer the next one.")
    parts.append("[Input]")
    parts.append(json.dumps(_example_to_kv(test_ex, label_key), ensure_ascii=False, indent=2))
    parts.append("[Answer]")
    return "\n".join(parts)


def pick_fewshot(
    train_split: List[Dict[str, Any]],
    n_shots: int,
    seed: int,
    label_key: str,
) -> List[Tuple[Dict[str, Any], Any]]:
    if n_shots <= 0 or not train_split:
        return []
    rnd = random.Random(seed)
    idxs = list(range(len(train_split)))
    rnd.shuffle(idxs)
    idxs = idxs[: min(n_shots, len(idxs))]
    out: List[Tuple[Dict[str, Any], Any]] = []
    for i in idxs:
        ex = train_split[i]
        out.append((ex, ex[label_key]))
    return out


def infer_label_key(example: Dict[str, Any]) -> Optional[str]:
    """Heuristic: LegalBench HF configs typically use 'label', but we check a few options."""
    for k in ["label", "answer", "output", "gold"]:
        if k in example:
            return k
    return None

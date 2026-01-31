import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple

@dataclass(frozen=True)
class TaskConfig:
    name: str
    # If provided, constrain answers to these canonical labels.
    labels: List[str]
    # Mapping from model output -> canonical label
    normalizers: List[Tuple[Pattern[str], str]]
    # A short instruction for the model.
    instruction: str

    def normalize(self, s: str) -> Optional[str]:
        s_clean = (s or "").strip()
        # Common cleanup: take first line, strip punctuation.
        s_clean = s_clean.splitlines()[0].strip()
        s_clean = re.sub(r"[\s\t]+", " ", s_clean)
        s_clean = s_clean.strip(" .,:;"'`()[]{}")
        low = s_clean.lower()

        # Exact match on canonical labels first (case-insensitive)
        for lab in self.labels:
            if low == lab.lower():
                return lab

        # Try regex normalizers
        for pat, lab in self.normalizers:
            if pat.search(low):
                return lab

        return None


# Recommended starter tasks (auto-gradable, mostly short outputs):
# - hearsay: binary classification (hearsay vs not hearsay)
# - personal_jurisdiction: binary classification (yes/no)
# - proa: binary classification (contains private right of action vs not)
# - privacy_policy_entailment: binary classification (Correct/Incorrect)
# - insurance_policy_interpretation: 3-way classification (Yes/No/Ambiguous)

TASKS: Dict[str, TaskConfig] = {
    "hearsay": TaskConfig(
        name="hearsay",
        labels=["Hearsay", "Not hearsay"],
        normalizers=[
            (re.compile(r"\bnot\s+hearsay\b"), "Not hearsay"),
            (re.compile(r"\bhearsay\b"), "Hearsay"),
            (re.compile(r"\b(inadmissible|out[- ]of[- ]court)\b"), "Hearsay"),
            (re.compile(r"\b(admissible|non[- ]hearsay)\b"), "Not hearsay"),
        ],
        instruction=(
            "Decide whether the evidence is hearsay under the provided definition. "
            "Answer with exactly one label: 'Hearsay' or 'Not hearsay'."
        ),
    ),
    "personal_jurisdiction": TaskConfig(
        name="personal_jurisdiction",
        labels=["Yes", "No"],
        normalizers=[
            (re.compile(r"\byes\b"), "Yes"),
            (re.compile(r"\bno\b"), "No"),
            (re.compile(r"\bhas\s+personal\s+jurisdiction\b"), "Yes"),
            (re.compile(r"\bno\s+personal\s+jurisdiction\b"), "No"),
        ],
        instruction=(
            "Determine if the forum court could exercise personal jurisdiction over the defendant. "
            "Answer with exactly: 'Yes' or 'No'."
        ),
    ),
    "proa": TaskConfig(
        name="proa",
        labels=["Yes", "No"],
        normalizers=[
            (re.compile(r"\byes\b"), "Yes"),
            (re.compile(r"\bno\b"), "No"),
            (re.compile(r"\bprivate\s+right\s+of\s+action\b"), "Yes"),
            (re.compile(r"\bno\s+private\s+right\s+of\s+action\b"), "No"),
        ],
        instruction=(
            "Decide whether the statute text contains an explicit private right of action. "
            "Answer with exactly: 'Yes' or 'No'."
        ),
    ),
    "privacy_policy_entailment": TaskConfig(
        name="privacy_policy_entailment",
        labels=["Correct", "Incorrect"],
        normalizers=[
            (re.compile(r"\bcorrect\b"), "Correct"),
            (re.compile(r"\bincorrect\b"), "Incorrect"),
            (re.compile(r"\b(entails|supported)\b"), "Correct"),
            (re.compile(r"\b(contradicts|not supported|does not entail)\b"), "Incorrect"),
        ],
        instruction=(
            "Given a privacy policy clause and a description, decide if the description is correct. "
            "Answer with exactly: 'Correct' or 'Incorrect'."
        ),
    ),
    "insurance_policy_interpretation": TaskConfig(
        name="insurance_policy_interpretation",
        labels=["A", "B", "C"],
        normalizers=[
            (re.compile(r"\ba\b"), "A"),
            (re.compile(r"\bb\b"), "B"),
            (re.compile(r"\bc\b"), "C"),
            (re.compile(r"\byes\b"), "A"),
            (re.compile(r"\bno\b"), "B"),
            (re.compile(r"\bambig|can't decide|cannot decide\b"), "C"),
        ],
        instruction=(
            "Read the insurance policy and claim. Choose: "
            "[A: Yes (covered); B: No (not covered); C: It's ambiguous]. "
            "Answer with exactly one of: A, B, or C."
        ),
    ),
}

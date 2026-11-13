"""Task metrics.

TextVQA uses the VQA accuracy: an answer gets min(1, matches / 3) against each
leave-one-out set of 9 human answers, averaged over the 10 sets, after the
answer normalization of the official VQA eval (punctuation, number words,
articles, common contractions).
"""

from __future__ import annotations

import re
from collections import defaultdict

PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
NUMBERS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
ARTICLES = {"a", "an", "the"}
# the official table is longer, these are the ones that actually show up in short answers
CONTRACTIONS = {
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "isnt": "isn't",
    "im": "i'm",
    "ive": "i've",
    "its": "it's",
    "lets": "let's",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
}


def _punctuation(text: str) -> str:
    out = text
    for p in PUNCT:
        if (p + " " in text or " " + p in text) or COMMA_STRIP.search(text) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return PERIOD_STRIP.sub("", out)


def _digits_articles(text: str) -> str:
    words = []
    for w in text.lower().split():
        w = NUMBERS.get(w, w)
        if w not in ARTICLES:
            words.append(CONTRACTIONS.get(w, w))
    return " ".join(words)


def normalize_vqa(answer: str) -> str:
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    return _digits_articles(_punctuation(answer))


def vqa_accuracy(pred: str, answers: list[str]) -> float:
    pred = normalize_vqa(pred)
    gts = [normalize_vqa(a) for a in answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1 :]
        accs.append(min(1.0, sum(g == pred for g in others) / 3))
    return sum(accs) / len(accs)


def yes_no(pred: str) -> str:
    """POPE parsing: first word if it's yes/no, else 'no' if a negation shows up, else 'yes'."""
    words = re.findall(r"[a-z']+", pred.lower())
    if words and words[0] in ("yes", "no"):
        return words[0]
    return "no" if any(w in ("no", "not") for w in words) else "yes"


def option_letter(pred: str) -> str | None:
    m = re.match(r"\s*\(?([A-Da-d])(?:[).:,\s]|$)", pred)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", pred)
    return m.group(1) if m else None


def pope(preds: list[str], answers: list[str]) -> dict:
    tp = fp = tn = fn = 0
    for p, a in zip(preds, answers, strict=True):
        y = yes_no(p)
        if a == "yes":
            tp += y == "yes"
            fn += y == "no"
        else:
            fp += y == "yes"
            tn += y == "no"
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": (tp + fp) / n if n else 0.0,
    }


def score(task: str, records: list[dict]) -> dict:
    """records: dicts with pred, answers and meta (from the eval jsonl)."""
    if task == "textvqa":
        accs = [vqa_accuracy(r["pred"], r["answers"]) for r in records]
        return {"n": len(accs), "accuracy": sum(accs) / len(accs) if accs else 0.0}
    if task == "pope":
        out = pope([r["pred"] for r in records], [r["answers"][0] for r in records])
        by_cat = defaultdict(list)
        for r in records:
            by_cat[r["meta"]["category"]].append(r)
        out["by_category"] = {
            c: pope([r["pred"] for r in rs], [r["answers"][0] for r in rs]) for c, rs in sorted(by_cat.items())
        }
        return out
    if task == "mmstar":
        hits = [option_letter(r["pred"]) == r["answers"][0] for r in records]
        by_cat = defaultdict(list)
        for r, h in zip(records, hits, strict=True):
            by_cat[r["meta"]["category"]].append(h)
        return {
            "n": len(hits),
            "accuracy": sum(hits) / len(hits) if hits else 0.0,
            "by_category": {c: sum(v) / len(v) for c, v in sorted(by_cat.items())},
        }
    raise ValueError(f"unknown task {task!r}")

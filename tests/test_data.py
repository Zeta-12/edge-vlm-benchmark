from edge_vlm.data import PROMPTS, Sample, subset


def make(n, cats=("a", "b", "c")):
    return [Sample(str(i), b"", f"q{i}", ["x"], {"task": "pope", "category": cats[i % len(cats)]}) for i in range(n)]


def test_subset_is_fixed_and_nested():
    s = make(100)
    a = [x.id for x in subset(s, 10, seed=0)]
    assert a == [x.id for x in subset(s, 10, seed=0)]
    assert a != [x.id for x in subset(s, 10, seed=1)]
    assert [x.id for x in subset(s, 30, seed=0)][:10] == a
    assert sorted(x.id for x in subset(s, None)) == sorted(x.id for x in s)


def test_stratified_prefixes_stay_balanced():
    picked = subset(make(90), 30, seed=0, stratify="category")
    for k in (3, 12, 30):
        cats = [p.meta["category"] for p in picked[:k]]
        assert all(cats.count(c) == k // 3 for c in "abc")
    assert len({p.id for p in picked}) == 30


def test_subset_stratified():
    picked = subset(make(90), 10, seed=0, stratify="category")
    counts = {c: sum(p.meta["category"] == c for p in picked) for c in "abc"}
    assert counts == {"a": 4, "b": 3, "c": 3}


def test_subset_bigger_than_set():
    assert len(subset(make(5), 10)) == 5
    assert len(subset(make(5), None)) == 5


def test_prompt():
    s = Sample("1", b"", "Is there a dog?", ["yes"], {"task": "pope"})
    assert s.prompt == "Is there a dog?\nAnswer the question using a single word or phrase."
    assert set(PROMPTS) == {"textvqa", "pope", "mmstar"}


def test_compare_agreement(tmp_path):
    import json

    from edge_vlm.compare import agreement

    def write(path, preds, tokens):
        pairs = enumerate(zip(preds, tokens, strict=True))
        rows = [{"id": str(i), "pred": p, "input_tokens": t} for i, (p, t) in pairs]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return path

    a = write(tmp_path / "a.jsonl", ["Exit", "the dog", "12"], [700, 800, 900])
    b = write(tmp_path / "b.jsonl", ["Exit", "dog", "13"], [700, 800, 901])
    res = agreement(a, b)
    assert (res["n"], res["same_text"], res["same_normalized"], res["same_input_tokens"]) == (3, 1, 2, 2)
    assert res["differences"] == [("2", "12", "13")]

import pytest

from edge_vlm import metrics


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("Dakota Digital.", "dakota digital"),
        ("The Coca-Cola", "coca cola"),
        ("two", "2"),
        ("1,000", "1000"),
        ("3.5", "3.5"),
        ("dont", "don't"),
        ("  yes\n", "yes"),
    ],
)
def test_normalize(raw, norm):
    assert metrics.normalize_vqa(raw) == norm


def test_vqa_accuracy_leave_one_out():
    answers = ["dakota"] * 6 + ["dakota digital"] * 3 + ["clos culombu"]
    assert metrics.vqa_accuracy("Dakota", answers) == 1.0
    # 3 matches: the 3 sets that drop one of them only have 2 left
    assert metrics.vqa_accuracy("dakota digital", answers) == pytest.approx((7 * 1 + 3 * 2 / 3) / 10)
    assert metrics.vqa_accuracy("clos culombu", answers) == pytest.approx(9 * (1 / 3) / 10)
    assert metrics.vqa_accuracy("nikon", answers) == 0.0


@pytest.mark.parametrize(
    ("pred", "label"),
    [("Yes", "yes"), ("No.", "no"), ("There is no dog.", "no"), ("It is not there", "no"), ("A dog is there", "yes")],
)
def test_yes_no(pred, label):
    assert metrics.yes_no(pred) == label


def test_pope_metrics():
    preds = ["yes", "yes", "no", "no", "yes"]
    answers = ["yes", "no", "no", "yes", "yes"]
    m = metrics.pope(preds, answers)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["yes_ratio"] == pytest.approx(3 / 5)


@pytest.mark.parametrize(
    ("pred", "letter"),
    [("B", "B"), ("(c) the cat", "C"), ("A. The suitcase", "A"), ("The answer is D", "D"), ("none", None)],
)
def test_option_letter(pred, letter):
    assert metrics.option_letter(pred) == letter


def test_score_by_category():
    recs = [
        {"pred": "yes", "answers": ["yes"], "meta": {"category": "random"}},
        {"pred": "yes", "answers": ["no"], "meta": {"category": "adversarial"}},
    ]
    out = metrics.score("pope", recs)
    assert out["accuracy"] == 0.5
    assert out["by_category"]["random"]["accuracy"] == 1.0
    assert out["by_category"]["adversarial"]["accuracy"] == 0.0

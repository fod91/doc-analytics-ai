from collections import Counter
from typing import Iterable, Tuple, Optional

Label = Optional[str]
Row = Tuple[str, Label]


def summarise_sentiment(rows: Iterable[Row]) -> dict:
    # Count labels across the dataset. Unknown/missing labels implies 'neu' (neutral)
    counts = Counter({"pos": 0, "neg": 0, "neu": 0})
    n = 0
    for _text, label in rows:
        n += 1
        lab = (label or "").strip().lower()
        if lab in ("pos", "positive", "+"):
            counts["pos"] += 1
        elif lab in ("neg", "negative", "-"):
            counts["neg"] += 1
        else:
            counts["neu"] += 1
    return {"n": n, "pos": counts["pos"], "neg": counts["neg"], "neu": counts["neu"]}

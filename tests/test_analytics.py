from doc_analytics_ai.analytics import summarise_sentiment


def test_summarise_sentiment_basic():
    rows = [
        ("a", "pos"),
        ("b", "neg"),
        ("c", None),
        ("d", "positive"),
        ("e", "negative"),
        ("f", "unknown"),
    ]
    out = summarise_sentiment(rows)
    assert out == {"n": 6, "pos": 2, "neg": 2, "neu": 2}

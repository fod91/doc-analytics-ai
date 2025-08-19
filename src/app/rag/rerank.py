from __future__ import annotations

import re
from typing import Dict, List, Iterable


PHRASE_BONUS = 1.0  # literal phrase match (or hyphen normalised)
TOKEN_BONUS = 0.25  # per query token with word boundary hit
EARLY_POS_MAX_BONUS = 0.10  # cap for "match near start" advantage
VEC_WEIGHT = 0.20  # how much normalised vector score contributes


_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    # Basic word tokenisation with Unicode support and case insensitive
    return [t for t in _WORD.findall((text or "").casefold()) if t]


def _phrase_variants(q: str) -> Iterable[str]:
    """
    Minimal normalisation: original and hyphen->space variant
    e.g., 'barrow-blade' -> {'barrow-blade', 'barrow blade'}.
    """
    q = (q or "").casefold().strip()
    if not q:
        return ()
    return {q, q.replace("-", " ")}


def keyword_score(query: str, text: str) -> float:
    """
    Lexical score made of:
      - PHRASE_BONUS if the query phrase (or its hyphenless variant) occurs
      - TOKEN_BONUS for each query token found at word boundaries
      - EARLY_POS_MAX_BONUS * proximity factor for phrase found near the start
    """
    t = (text or "").casefold()
    score = 0.0

    # full phrase containment
    positions = []
    variants = list(_phrase_variants(query))
    for v in variants:
        if v and v in t:
            score += PHRASE_BONUS
            positions.append(t.find(v))

    # token overlap
    for tok in _tokens(query):
        if re.search(rf"\b{re.escape(tok)}\b", t):
            score += TOKEN_BONUS

    # early position bonus
    if positions:
        pos = min(positions)
        L = max(1, len(t))
        proximity = (L - pos) / L  # ~=1 near thestart; ~=0 near the end
        score += EARLY_POS_MAX_BONUS * proximity

    return score


def rerank_keyword(query: str, contexts: List[Dict]) -> List[Dict]:
    """
    Combine keyword score with a vector score to stabilise ties.
    Assumes contexts include 'text', 'score', and 'rank' (original)
    """
    if not contexts:
        return contexts

    # normalise vector scores roughly into [0, 1]
    vs = [float(c.get("score", 0.0)) for c in contexts]
    vmin, vmax = min(vs), max(vs)
    span = (vmax - vmin) or 1.0

    def combined(c: Dict) -> float:
        kw = keyword_score(query, c.get("text", ""))
        v = (float(c.get("score", 0.0)) - vmin) / span
        return kw + VEC_WEIGHT * v

    # sort by combined desc
    # for a tie-break preserve earlier vector rank
    return sorted(
        contexts, key=lambda c: (combined(c), -int(c.get("rank", 0))), reverse=True
    )

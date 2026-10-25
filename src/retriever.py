"""BM25 keyword retriever over KB chunks (no extra dependencies)."""
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[\w\u0900-\u097F]+")      # Latin, digits and the whole Devanagari block
_STOP = set("""the a an is are was were to of for on in my me i can do does how what it if and or be by will
with at this that you your am get got have has had so not no from as there their then than
hai ka ki ke mein kya ko se pe par kab tak kaise wala wali hua hui hoon sakti sakta aa jaana chahiye karein
karna kar mera meri mujhe aap hum""".split())

# lower = less preferred when two chunks score similarly (mirrors KB-00.1 precedence)
_TYPE_WEIGHT = {"policy": 1.0, "agent_guide": 0.9, "help": 0.9, "faq": 0.85, "marketing": 0.5, "meta": 0.8}


def tokenize(text: str) -> list:
    out = []
    for t in _TOKEN.findall(str(text).lower()):
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]                         # crude plural stemming: refunds -> refund
        if t not in _STOP:
            out.append(t)
    return out


class BM25Retriever:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c.search_text())) for c in chunks]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, q_tokens, i):
        d, dl, s = self.docs[i], self.len[i], 0.0
        for t in q_tokens:
            f = d.get(t, 0)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s

    def search(self, query: str, k: int = 6, exclude_ids=()):
        q = tokenize(query)
        scored = []
        for i, c in enumerate(self.chunks):
            if c.id in exclude_ids:
                continue
            s = self._score(q, i) * _TYPE_WEIGHT.get(c.doc_type, 1.0)
            if c.status != "current":
                s *= 0.7
            if s > 0:
                scored.append((c, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

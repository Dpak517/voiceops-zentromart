"""Find product and pincode records mentioned in a customer message."""
import csv
import os
import re

from .datatools import DATA_DIR

_SKU = re.compile(r"\b[A-Z]{3}-\d{4}\b")
_PIN = re.compile(r"(?<!\d)\d{6}(?!\d)")
_STOP = {"the", "a", "an", "of", "and", "with", "for", "pack", "set"}


def _read(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


class Records:
    def __init__(self):
        self.products = _read("products.csv")
        self.by_sku = {p["sku"]: p for p in self.products}
        self.pins = {p["pincode"]: p for p in _read("pincodes.csv")}
        self._title_tokens = {p["sku"]: {t for t in _norm(p["title"]).split() if t not in _STOP}
                              for p in self.products}

    def find_products(self, text: str, max_items=3) -> list:
        found = {}
        for sku in _SKU.findall(text.upper()):
            if sku in self.by_sku:
                found[sku] = self.by_sku[sku]
        q = _norm(text)
        for p in self.products:                              # exact title mentioned
            if _norm(p["title"]) in q:
                found.setdefault(p["sku"], p)
        if not found:                                        # fuzzy: most title words present
            qt, scored = set(q.split()), []
            for p in self.products:
                tt = self._title_tokens[p["sku"]]
                if len(tt) >= 2:
                    ov = len(tt & qt) / len(tt)
                    if ov >= 0.8:
                        scored.append((ov, p))
            scored.sort(key=lambda x: -x[0])
            for _, p in scored[:max_items]:
                found.setdefault(p["sku"], p)
        return list(found.values())[:max_items]

    def find_pincodes(self, text: str) -> list:
        return [self.pins[p] for p in _PIN.findall(text) if p in self.pins]

    @staticmethod
    def render_product(p) -> str:
        return (f"[products.csv:{p['sku']}] title={p['title']}; brand={p['brand']}; "
                f"return_category={p['return_category']}; price_inr={p['price_inr']}; "
                f"fulfilled_by={p['fulfilled_by']}; assured={p['assured']}; warranty_months={p['warranty_months']}")

    @staticmethod
    def render_pincode(p) -> str:
        return (f"[pincodes.csv:{p['pincode']}] city={p['city']}; serviceable={p['serviceable']}; "
                f"cod_available={p['cod_available']}; express_available={p['express_available']}")

"""Query expansion: bridge customer wording -> document wording (e.g. 'send back' -> 'return').

This only adds SEARCH keywords. It never contains answers; the answers come from retrieved chunks."""
import re

# (regex over the lower-cased question, keywords to add to the search query)
_RULES = [
    (r"send (it )?back|sending back|give back|take back|collection|collect|return|exchange|wapas|लौटा|वापस|लौटाने",
     "return window days policy reverse pickup fee category"),
    (r"doesn.?t fit|don.?t like|not like|size|change of mind|pasand", "change of mind return reverse pickup fee"),
    (r"open(ed)?|used|tried|unseal|seal|खोल", "return condition unopened sealed unused beauty"),
    (r"refund|money back|paid back|reimburse|पैसे|रिफ़ंड|रिफंड|paisa",
     "refund timeline business days payment method initiated"),
    (r"by which date|which date|how soon|how long|how fast|kab|कब|kitne din|कितने दिन|when will",
     "timeline days business days holidays date"),
    (r"cash on delivery|\bcod\b|कैश", "cash on delivery refund wallet bank transfer"),
    (r"crack|broke|broken|torn|damag|defect|wrong item|missing|टूट|खराब|गलत|came torn|expired",
     "damaged defective wrong missing report 48 hours replacement refund high-value priority"),
    (r"not home|not at home|nobody|not available|reschedul|courier comes|अनुपस्थित", "delivery attempts reschedule return to origin"),
    (r"already (on its way|shipped)|on its way|shipped|शिप|stop it|stop my order|cancel|कैंसिल",
     "cancel shipped order refuse parcel before ships cancellation"),
    (r"warranty|assured|guarantee|claim", "warranty assured brand warranty months claim marketplace seller"),
    (r"never placed|didn.?t place|did not place|not mine|unauthori[sz]ed|hacked|fraud|scam|stolen|without my",
     "order payment account activity customer did not make escalation human handover"),
    (r"phone number|address|email|details of (the )?(person|customer)|another customer|someone else",
     "never share other customer details verification phone address"),
    (r"\blate\b|delayed|arrived a day|देर", "late delivery express fee credit wallet"),
    (r"\bsale\b|festive|offer|सेल", "festive sale returns clothing footwear bags 30 days"),
    (r"deliver(y|ed)? to|ship to|dubai|abroad|outside india|nepal|usa|international|kochi|कोच्चि|store|pick ?up from",
     "where we deliver serviceable pincode India stores"),
    (r"coupon|discount|promo", "coupon discount refund partial"),
    (r"plus|member", "ZentroMart Plus membership benefits"),
    (r"human|agent|person|colleague|legal|court|police|consumer", "escalation human handover legal"),
]


_DAMAGE = r"crack|broke|broken|torn|damag|defect|wrong item|missing|टूट|खराब|गलत|expired"


def expand_query(question: str, products=(), pincodes=(), code_labels=None) -> str:
    q = question.lower()
    extra = [kw for pat, kw in _RULES if re.search(pat, q)]
    damage = bool(re.search(_DAMAGE, q))
    for p in products:
        code = p["return_category"]
        extra.append(code.replace("_", " "))                              # e.g. 'innerwear', 'electronics'
        if code_labels and code in code_labels:
            extra.append(code_labels[code])                               # label parsed from RET-2
        if str(p.get("fulfilled_by", "")).lower() != "zentromart":
            extra.append("marketplace seller")
        if p.get("assured") == "True":
            extra.append("ZentroMart Assured")
        if damage and float(p.get("price_inr") or 0) > 10000:
            extra.append("high-value more than 10000 Priority Resolution")
    if pincodes:
        extra.append("serviceable pincode cash on delivery express")
    return question + " " + " ".join(extra)

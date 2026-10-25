"""Date calculators the LLM can call. They do arithmetic only: the policy numbers
(e.g. '5 to 7 business days', '15 day window') must come from the retrieved documents."""
import csv
import os
from datetime import date, datetime, timedelta

from .kb import ROOT

DATA_DIR = os.path.join(ROOT, "data")
TODAY = date(2026, 10, 20)       # KB-00.2: the dataset is frozen on 20 October 2026


def _holidays():
    with open(os.path.join(DATA_DIR, "holidays.csv"), encoding="utf-8", newline="") as f:
        return {datetime.strptime(r["date"], "%Y-%m-%d").date() for r in csv.DictReader(f)}


def _fmt(d: date) -> dict:
    return {"date": d.isoformat(), "formatted": f"{d.day} {d.strftime('%B %Y')}", "weekday": d.strftime("%A")}


def add_calendar_days(start_date: str, days: int) -> dict:
    d = datetime.strptime(start_date, "%Y-%m-%d").date() + timedelta(days=int(days))
    return _fmt(d)


def add_business_days(start_date: str, days: int) -> dict:
    """N business days AFTER start_date (start_date itself is not counted).
    Business day = Monday to Saturday, excluding holidays.csv (KB-00.3)."""
    hol, d, left = _holidays(), datetime.strptime(start_date, "%Y-%m-%d").date(), int(days)
    if left < 0:
        raise ValueError("days must be >= 0")
    while left:
        d += timedelta(days=1)
        if d.weekday() != 6 and d not in hol:
            left -= 1
    return _fmt(d)


def business_day_range(start_date: str, min_days: int, max_days: int) -> dict:
    """Earliest and latest date for a 'MIN to MAX business days' timeline, in ONE call."""
    return {"earliest": add_business_days(start_date, min_days),
            "latest": add_business_days(start_date, max_days)}


def business_days_between(start_date: str, end_date: str) -> dict:
    """How many business days have passed: start_date is not counted, end_date is.
    Business day = Monday-Saturday excluding holidays.csv. Used to check if a refund is overdue."""
    hol = _holidays()
    d, end = datetime.strptime(start_date, "%Y-%m-%d").date(), datetime.strptime(end_date, "%Y-%m-%d").date()
    n = 0
    while d < end:
        d += timedelta(days=1)
        if d.weekday() != 6 and d not in hol:
            n += 1
    return {"business_days": n}


def days_between(start_date: str, end_date: str) -> dict:
    a = datetime.strptime(start_date, "%Y-%m-%d").date()
    b = datetime.strptime(end_date, "%Y-%m-%d").date()
    return {"calendar_days": (b - a).days}


TOOL_IMPLS = {"add_calendar_days": add_calendar_days, "add_business_days": add_business_days,
              "business_day_range": business_day_range, "business_days_between": business_days_between,
              "days_between": days_between}

_DATE = {"type": "string", "description": "Date as YYYY-MM-DD"}
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "business_day_range",
        "description": "For a timeline of 'MIN to MAX business days' (e.g. 5 to 7): returns BOTH the earliest and "
                       "the latest date in one call. Start date is not counted; Sundays and holidays are skipped.",
        "parameters": {"type": "object", "properties": {"start_date": _DATE, "min_days": {"type": "integer"},
                                                        "max_days": {"type": "integer"}},
                       "required": ["start_date", "min_days", "max_days"]}}},
    {"type": "function", "function": {
        "name": "add_business_days",
        "description": "Date that is N business days after start_date (start date not counted). "
                       "Business days = Monday-Saturday excluding warehouse holidays. Use for refund and logistics timelines.",
        "parameters": {"type": "object", "properties": {"start_date": _DATE, "days": {"type": "integer"}},
                       "required": ["start_date", "days"]}}},
    {"type": "function", "function": {
        "name": "add_calendar_days",
        "description": "Date that is N calendar days after start_date. Use for return-window deadlines.",
        "parameters": {"type": "object", "properties": {"start_date": _DATE, "days": {"type": "integer"}},
                       "required": ["start_date", "days"]}}},
    {"type": "function", "function": {
        "name": "business_days_between",
        "description": "Number of business days that have passed from start_date to end_date (start not counted, "
                       "end counted; Sundays and holidays skipped). Use it to see how long a refund has been pending.",
        "parameters": {"type": "object", "properties": {"start_date": _DATE, "end_date": _DATE},
                       "required": ["start_date", "end_date"]}}},
    {"type": "function", "function": {
        "name": "days_between",
        "description": "Number of calendar days from start_date to end_date.",
        "parameters": {"type": "object", "properties": {"start_date": _DATE, "end_date": _DATE},
                       "required": ["start_date", "end_date"]}}},
]

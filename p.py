#!/usr/bin/env python3
"""
Library Circulation Trends Over Time (Seattle Public Library)

What it makes:
1) Line chart of checkouts by genre over time
2) Heatmap of seasonal reading trends (month x year)
3) Small multiples by branch/location (if the dataset provides a branch/location column)

Data source:
- City of Seattle Open Data (Socrata / SODA API)
  - Checkouts by Title (tmmm-ytt6): monthly count by title (physical + electronic)
  - Optional: Checkouts by Title (Physical Items) (5src-czff): checkout log, often more granular

Docs:
- Socrata SODA: https://dev.socrata.com/
"""

from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# Config
# ----------------------------

DEFAULT_DOMAIN = "data.seattle.gov"

# Monthly counts by title (physical + electronic). Good for time series and COVID-era story.
DEFAULT_DATASET_ID = "tmmm-ytt6"

# Optional: physical checkout log (can be huge). Uncomment if you specifically want it.
# DEFAULT_DATASET_ID = "5src-czff"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_context("talk")


@dataclass
class SocrataConfig:
    domain: str = DEFAULT_DOMAIN
    dataset_id: str = DEFAULT_DATASET_ID
    app_token: Optional[str] = None  # set env SOCRATA_APP_TOKEN if you have one
    page_size: int = 50_000
    max_rows: Optional[int] = 500_000  # cap for safety; set None to fetch all


# ----------------------------
# Socrata fetch
# ----------------------------

def fetch_socrata_all(cfg: SocrataConfig, where: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch rows from a Socrata dataset using pagination ($limit/$offset).
    Returns a DataFrame.
    """
    base_url = f"https://{cfg.domain}/resource/{cfg.dataset_id}.json"

    headers = {}
    if cfg.app_token:
        headers["X-App-Token"] = cfg.app_token

    rows: list[dict] = []
    offset = 0
    fetched = 0

    while True:
        params = {
            "$limit": cfg.page_size,
            "$offset": offset,
        }
        if where:
            params["$where"] = where

        resp = requests.get(base_url, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        chunk = resp.json()

        if not chunk:
            break

        rows.extend(chunk)
        fetched += len(chunk)
        offset += cfg.page_size

        if cfg.max_rows is not None and fetched >= cfg.max_rows:
            break

    df = pd.DataFrame(rows)
    return df


# ----------------------------
# Column helpers (robust to naming differences)
# ----------------------------

def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Pick the first matching column from candidates (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def to_int_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


# ----------------------------
# Genre inference (rule-based)
# ----------------------------

GENRE_RULES = [
    ("Children",  [r"\bjuvenile\b", r"\bchild", r"\bkids?\b", r"\bpicture book\b"]),
    ("Young Adult",[r"\bya\b", r"\byoung adult\b", r"\bteen\b"]),
    ("Mystery",   [r"\bmystery\b", r"\bdetective\b", r"\bthriller\b", r"\bcrime\b"]),
    ("Romance",   [r"\bromance\b", r"\blove story\b"]),
    ("Sci-Fi",    [r"\bsci[- ]?fi\b", r"\bscience fiction\b", r"\bspace\b", r"\bcyberpunk\b"]),
    ("Fantasy",   [r"\bfantasy\b", r"\bdragon\b", r"\bmagic\b"]),
    ("Horror",    [r"\bhorror\b", r"\bghost\b", r"\bvampire\b", r"\boccult\b"]),
    ("Comics",    [r"\bgraphic novel\b", r"\bcomic\b", r"\bmanga\b"]),
    ("History",   [r"\bhistory\b", r"\bbiography\b", r"\bworld war\b"]),
    ("Business",  [r"\bbusiness\b", r"\beconomics\b", r"\bfinance\b", r"\binvesting\b"]),
    ("Health",    [r"\bhealth\b", r"\bmedicine\b", r"\bnutrition\b", r"\bmental health\b"]),
    ("Self-Help", [r"\bself-help\b", r"\bself help\b", r"\bpersonal development\b"]),
    ("Cooking",   [r"\bcook", r"\brecipes?\b", r"\bbaking\b"]),
    ("Travel",    [r"\btravel\b", r"\bguid(e|book)\b"]),
    ("Religion",  [r"\breligion\b", r"\bspiritual\b", r"\btheology\b"]),
    ("Poetry",    [r"\bpoetry\b", r"\bpoems?\b"]),
]

def infer_genre(text: str) -> str:
    t = (text or "").lower()
    for genre, patterns in GENRE_RULES:
        for p in patterns:
            if re.search(p, t):
                return genre
    # If nothing matches, keep it broad
    return "General / Other"


# ----------------------------
# Preprocess
# ----------------------------

def preprocess_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the monthly dataset (tmmm-ytt6-like):
    - builds a month-start datetime column
    - creates a 'genre' column inferred from subjects/title/material
    - ensures numeric checkouts
    """
    year_col = pick_col(df, ["CheckoutYear", "checkoutyear", "year"])
    month_col = pick_col(df, ["CheckoutMonth", "checkoutmonth", "month"])
    checkouts_col = pick_col(df, ["Checkouts", "checkouts"])

    title_col = pick_col(df, ["Title", "title"])
    subjects_col = pick_col(df, ["Subjects", "subjects", "Subject", "subject"])
    material_col = pick_col(df, ["MaterialType", "materialtype", "ItemType", "itemtype"])
    checkout_type_col = pick_col(df, ["CheckoutType", "checkouttype"])  # physical vs electronic (often)

    if year_col is None or month_col is None or checkouts_col is None:
        raise ValueError(
            "Could not find required time/checkouts columns. "
            f"Found columns: {list(df.columns)[:30]} ..."
        )

    out = df.copy()
    out[year_col] = to_int_safe(out[year_col])
    out[month_col] = to_int_safe(out[month_col])
    out[checkouts_col] = pd.to_numeric(out[checkouts_col], errors="coerce").fillna(0)

    out = out.dropna(subset=[year_col, month_col])
    out["date"] = pd.to_datetime(
        dict(year=out[year_col].astype(int), month=out[month_col].astype(int), day=1),
        errors="coerce",
    )
    out = out.dropna(subset=["date"])

    # Build a text blob for genre inference
    text_parts = []
    for col in [subjects_col, title_col, material_col]:
        if col is not None:
            text_parts.append(out[col].astype(str))
    if text_parts:
        blob = text_parts[0]
        for part in text_parts[1:]:
            blob = blob + " | " + part
        out["genre"] = blob.apply(infer_genre)
    else:
        out["genre"] = "General / Other"

    # Keep a simplified column name set
    out = out.rename(columns={
        checkouts_col: "checkouts",
        year_col: "year",
        month_col: "month",
    })

    if checkout_type_col is not None:
        out = out.rename(columns={checkout_type_col: "checkout_type"})
    else:
        out["checkout_type"] = "Unknown"

    return out


# ----------------------------
# Visualizations
# ----------------------------

def plot_genre_lines(df: pd.DataFrame, top_n: int = 8) -> str:
    """
    Line chart: monthly checkouts by genre (top N genres overall).
    Saves to output/genre_trends.png
    """
    g = (df.groupby(["date", "genre"], as_index=False)["checkouts"].sum())
    top_genres = (g.groupby("genre")["checkouts"].sum().sort_values(ascending=False).head(top_n).index)
    g = g[g["genre"].isin(top_genres)].copy()

    plt.figure(figsize=(14, 7))
    for genre, sub in g.groupby("genre"):
        sub = sub.sort_values("date")
        plt.plot(sub["date"], sub["checkouts"], label=genre)

    # Story marker: COVID period shading (approx.)
    plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.15)
    plt.title("Library Checkouts by Genre Over Time (Top Genres)\n(shaded: early COVID era ~ Mar 2020–Jun 2021)")
    plt.xlabel("Date")
    plt.ylabel("Monthly Checkouts")
    plt.legend(ncol=2, frameon=True)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "genre_trends.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_seasonality_heatmap(df: pd.DataFrame) -> str:
    """
    Heatmap: month-of-year vs year of total checkouts.
    Saves to output/seasonality_heatmap.png
    """
    h = (df.groupby(["year", "month"], as_index=False)["checkouts"].sum())
    pivot = h.pivot(index="year", columns="month", values="checkouts").sort_index()

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, cmap="viridis", linewidths=0.2)
    plt.title("Seasonal Reading Trends (Total Checkouts)\nYear × Month")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "seasonality_heatmap.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_small_multiples_by_branch(df: pd.DataFrame, branch_col: str, top_n: int = 8) -> Optional[str]:
    """
    Small multiples by branch/location if available.
    Saves to output/small_multiples_branch.png
    """
    s = df.copy()
    s[branch_col] = s[branch_col].astype(str)

    totals = s.groupby(branch_col)["checkouts"].sum().sort_values(ascending=False)
    top_branches = totals.head(top_n).index
    s = s[s[branch_col].isin(top_branches)]

    agg = s.groupby(["date", branch_col], as_index=False)["checkouts"].sum()

    g = sns.FacetGrid(agg, col=branch_col, col_wrap=2, height=3.2, sharey=False)
    g.map_dataframe(sns.lineplot, x="date", y="checkouts")
    g.set_axis_labels("Date", "Monthly Checkouts")
    g.set_titles(col_template="{col_name}")
    g.fig.suptitle("Small Multiples: Checkout Trends by Branch/Location (Top Locations)", y=1.02)
    g.tight_layout()

    path = os.path.join(OUTPUT_DIR, "small_multiples_branch.png")
    g.savefig(path, dpi=200)
    plt.close("all")
    return path


# ----------------------------
# Main
# ----------------------------

def main():
    cfg = SocrataConfig(
        dataset_id=os.getenv("SOCRATA_DATASET_ID", DEFAULT_DATASET_ID),
        app_token=os.getenv("SOCRATA_APP_TOKEN"),
        max_rows=int(os.getenv("SOCRATA_MAX_ROWS", "500000")),
    )

    print(f"Fetching dataset {cfg.dataset_id} from {cfg.domain} ...")
    raw = fetch_socrata_all(cfg)
    print(f"Fetched rows: {len(raw):,}")
    print(f"Columns: {list(raw.columns)}")

    # Preprocess (assumes monthly dataset structure; works best with tmmm-ytt6)
    df = preprocess_monthly(raw)
    print("Date range:", df["date"].min(), "→", df["date"].max())
    print("Sample genres:", df["genre"].value_counts().head(10).to_dict())

    # Visuals
    p1 = plot_genre_lines(df, top_n=8)
    p2 = plot_seasonality_heatmap(df)

    # Try to find a branch/location-ish column for small multiples.
    branch_col = pick_col(df, ["ItemLocation", "itemlocation", "Branch", "branch", "Location", "location"])
    p3 = None
    if branch_col is not None:
        p3 = plot_small_multiples_by_branch(df, branch_col=branch_col, top_n=8)

    print("\nSaved outputs:")
    print(" -", p1)
    print(" -", p2)
    if p3:
        print(" -", p3)
    else:
        print(" - (No branch/location column found; small multiples skipped.)")

    # Bonus: a quick COVID physical vs digital story if checkout_type exists
    if "checkout_type" in df.columns and df["checkout_type"].nunique() > 1:
        covid = df[(df["date"] >= "2018-01-01")].copy()
        trend = covid.groupby(["date", "checkout_type"], as_index=False)["checkouts"].sum()

        plt.figure(figsize=(14, 7))
        for t, sub in trend.groupby("checkout_type"):
            sub = sub.sort_values("date")
            plt.plot(sub["date"], sub["checkouts"], label=t)
        plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.15)
        plt.title("Physical vs Digital Borrowing Over Time\n(shaded: early COVID era ~ Mar 2020–Jun 2021)")
        plt.xlabel("Date")
        plt.ylabel("Monthly Checkouts")
        plt.legend(frameon=True)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "physical_vs_digital.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(" -", path)


if __name__ == "__main__":
    main()

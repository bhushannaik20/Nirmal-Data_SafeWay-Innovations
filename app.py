
import io
import json
import math
from dataclasses import dataclass
from html import escape
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer, KNNImputer

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

APP_TITLE = "CleanSheet v3 — Data Cleaning • Weighting • PDF Report"
APP_DESC = "Upload CSV/Excel → Schema map (UI/JSON) → Impute → Outliers → Validate → Weighted & Unweighted summaries → Visual diagnostics → HTML/PDF report"

# ---------- utils ----------
def bytes_from_df_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def bytes_from_df_excel(df: pd.DataFrame, sheet_name="Cleaned"):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    bio.seek(0)
    return bio.getvalue()

def memory_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / (1024*1024)

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = (miss / len(df))*100 if len(df) else 0
    uniq = df.nunique(dropna=True)
    return pd.DataFrame({"missing": miss, "missing_%": np.round(pct, 2), "unique": uniq}).sort_values("missing", ascending=False)

def weighted_mean_and_se(x: np.ndarray, w: np.ndarray):
    W = np.sum(w)
    if W <= 0:
        return (np.nan, np.nan)
    mu = np.sum(w * x) / W
    var = np.sum((w**2) * (x - mu) ** 2) / (W**2)
    se = np.sqrt(var)
    return float(mu), float(se)

def weighted_prop_and_se(y: np.ndarray, w: np.ndarray):
    W = np.sum(w)
    if W <= 0:
        return (np.nan, np.nan)
    p = np.sum(w * y) / W
    var = np.sum((w**2) * (y - p) ** 2) / (W**2)
    se = np.sqrt(var)
    return float(p), float(se)

def margin_of_error_95(se: float):
    if se is None or (isinstance(se, float) and (math.isnan(se))):
        return np.nan
    return 1.96 * se

# ---------- models ----------
@dataclass
class ValidationRule:
    rule_type: str
    params: dict

@dataclass
class PipelineConfig:
    weight_column: Optional[str] = None
    columns: Dict[str, dict] = None
    validations: List[ValidationRule] = None

# ---------- validation ----------
def validate_rules(df: pd.DataFrame, rules: List[ValidationRule]) -> pd.DataFrame:
    violations = []
    for idx, rule in enumerate(rules or []):
        rtype = rule.rule_type
        p = rule.params or {}
        try:
            if rtype == "range":
                col = p["column"]; mn = p.get("min", -np.inf); mx = p.get("max", np.inf)
                bad_idx = df.index[(pd.to_numeric(df[col], errors="coerce") < mn) | (pd.to_numeric(df[col], errors="coerce") > mx)]
                for i in bad_idx: violations.append((i, rtype, f"{col} out of [{mn}, {mx}]"))
            elif rtype == "allowed_values":
                col = p["column"]; vals = set(p.get("values", []))
                bad_idx = df.index[~df[col].isin(vals)]
                for i in bad_idx: violations.append((i, rtype, f"{col} not in {sorted(list(vals))}"))
            elif rtype == "regex":
                import re
                col = p["column"]; pattern = re.compile(p["pattern"])
                bad_idx = df.index[~df[col].astype(str).str.match(pattern)]
                for i in bad_idx: violations.append((i, rtype, f"{col} fails regex"))
            elif rtype == "cross_field":
                cond = p.get("if", {}); then = p.get("then", {})
                ccol = cond.get("column"); ceq = cond.get("equals")
                tcol = then.get("column"); must_not_null = then.get("not_null", False)
                must_be_null = then.get("must_be_null", False)
                mask = df[ccol] == ceq
                if must_not_null:
                    bad_idx = df.index[mask & (df[tcol].isna() | (df[tcol].astype(str).str.len() == 0))]
                    for i in bad_idx: violations.append((i, rtype, f"When {ccol}=={ceq}, {tcol} must be non-null"))
                if must_be_null:
                    bad_idx = df.index[mask & (~df[tcol].isna()) & (df[tcol].astype(str).str.len() > 0)]
                    for i in bad_idx: violations.append((i, rtype, f"When {ccol}=={ceq}, {tcol} must be null"))
            elif rtype == "skip_pattern":
                cond = p.get("if", {}); then = p.get("then", {})
                ccol = cond.get("column"); ceq = cond.get("equals")
                tcol = then.get("column"); must_be_null = then.get("must_be_null", True)
                mask = df[ccol] == ceq
                if must_be_null:
                    bad_idx = df.index[mask & (~df[tcol].isna()) & (df[tcol].astype(str).str.len() > 0)]
                    for i in bad_idx: violations.append((i, rtype, f"Skip: If {ccol}=={ceq}, {tcol} must be null"))
        except Exception as e:
            st.warning(f"Validation rule {idx+1} failed: {e}")
    return pd.DataFrame(violations, columns=["row_index","rule_type","message"]) if violations else pd.DataFrame(columns=["row_index","rule_type","message"])

# ---------- cleaning ----------
def impute_missing(df: pd.DataFrame, config: PipelineConfig, logs: List[str]) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    default_num = st.session_state.get('default_numeric_impute', '(none)')
    default_cat = st.session_state.get('default_categorical_impute', '(none)')
    default_const_num = st.session_state.get('default_constant_numeric', 0.0)
    default_const_cat = st.session_state.get('default_constant_categorical', '')

    knn_cols = []
    for col, cconf in (config.columns or {}).items():
        if isinstance(cconf, dict):
            strategy = (cconf.get('impute') or {}).get('strategy')
            if strategy == 'knn' and col in numeric_cols:
                knn_cols.append(col)

    for col, cconf in (config.columns or {}).items():
        if not isinstance(cconf, dict): continue
        imp = cconf.get('impute', {}); strategy = imp.get('strategy')
        if not strategy or strategy == 'none': continue
        if strategy in ['mean','median','most_frequent','constant'] and col in df.columns:
            if strategy == 'most_frequent':
                imputer = SimpleImputer(strategy='most_frequent')
            elif strategy == 'constant':
                imputer = SimpleImputer(strategy='constant', fill_value=imp.get('constant', 0))
            else:
                imputer = SimpleImputer(strategy=strategy)
            try:
                df[[col]] = imputer.fit_transform(df[[col]])
                logs.append(f"Imputed {col} using {strategy}")
            except Exception as e:
                logs.append(f"Imputation failed for {col} with {strategy}: {e}")

    for col in df.columns:
        if df[col].isna().any():
            cconf = (config.columns or {}).get(col, {}) if isinstance((config.columns or {}), dict) else {}
            has_explicit = isinstance(cconf, dict) and (cconf.get('impute') not in [None, {}, {'strategy':'none'}])
            if not has_explicit:
                is_num = col in numeric_cols
                strat = default_num if is_num else default_cat
                if strat and strat != '(none)':
                    if strat == 'constant':
                        fill_value = default_const_num if is_num else default_const_cat
                        imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                    else:
                        imputer = SimpleImputer(strategy=strat)
                    try:
                        df[[col]] = imputer.fit_transform(df[[col]])
                        logs.append(f"Imputed {col} using DEFAULT {strat}")
                    except Exception as e:
                        logs.append(f"DEFAULT imputation failed for {col}: {e}")

    if knn_cols:
        try:
            k = int(st.session_state.get("knn_k", 5))
        except:
            k = 5
        try:
            imputer = KNNImputer(n_neighbors=k)
            df[knn_cols] = imputer.fit_transform(df[knn_cols])
            logs.append(f"KNN-imputed cols {knn_cols} with k={k}")
        except Exception as e:
            logs.append(f"KNN Imputation failed for {knn_cols}: {e}")
    return df

def zscore_outliers(s: pd.Series, z_thresh: float):
    m = s.mean(); sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd): return pd.Series([False]*len(s), index=s.index)
    z = (s - m) / sd
    return z.abs() > z_thresh

def iqr_outliers(s: pd.Series, k: float = 1.5):
    q1 = s.quantile(0.25); q3 = s.quantile(0.75)
    iqr = q3 - q1; lo = q1 - k*iqr; hi = q3 + k*iqr
    return (s < lo) | (s > hi)

def winsorize_series(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    lo = s.quantile(lower_q); hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

def handle_outliers(df: pd.DataFrame, config: PipelineConfig, logs: List[str]):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_flags = pd.DataFrame(index=df.index)
    for col, cconf in (config.columns or {}).items():
        if col not in numeric_cols: continue
        oc = (cconf or {}).get("outliers", {})
        method = oc.get("method"); action = oc.get("action", "flag"); params = oc.get("params", {})
        if not method: continue
        try:
            if method == "zscore":
                z = float(params.get("z_thresh", 3.0)); mask = zscore_outliers(df[col].astype(float), z)
            elif method == "iqr":
                k = float(params.get("k", 1.5)); mask = iqr_outliers(df[col].astype(float), k)
            elif method == "winsorize":
                lq = float(params.get("lower_q", 0.01)); uq = float(params.get("upper_q", 0.99))
                lo = df[col].quantile(lq); hi = df[col].quantile(uq); mask = (df[col] < lo) | (df[col] > hi)
            else:
                continue
            outlier_flags[col] = mask
            if action == "flag":
                logs.append(f"Outliers flagged in {col} using {method}")
            elif action == "remove":
                n_before = len(df); df = df.loc[~mask].copy(); n_after = len(df)
                logs.append(f"Removed {n_before - n_after} outliers in {col} using {method}")
            elif action == "clip":
                if method == "winsorize":
                    lq = float(params.get("lower_q", 0.01)); uq = float(params.get("upper_q", 0.99))
                    df[col] = winsorize_series(df[col].astype(float), lq, uq)
                    logs.append(f"Winsorized {col} to [{lq*100:.1f}%, {uq*100:.1f}%]")
                else:
                    if method == "zscore":
                        z = float(params.get("z_thresh", 3.0))
                        m = df[col].mean(); sd = df[col].std(ddof=0); lo = m - z*sd; hi = m + z*sd
                    else:
                        q1 = df[col].quantile(0.25); q3 = df[col].quantile(0.75); iqr = q3-q1
                        lo = q1 - float(params.get("k",1.5))*iqr; hi = q3 + float(params.get("k",1.5))*iqr
                    df[col] = df[col].clip(lower=lo, upper=hi)
                    logs.append(f"Clipped {col} to [{lo:.4g}, {hi:.4g}] using {method}")
        except Exception as e:
            logs.append(f"Outlier handling failed for {col}: {e}")
    outlier_flags = outlier_flags.loc[df.index] if len(outlier_flags) else pd.DataFrame(index=df.index)
    return df, outlier_flags

# ---------- summaries ----------
def compute_summaries(df: pd.DataFrame, weight_col: Optional[str]) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if weight_col and weight_col in numeric_cols:
        numeric_cols_wo_w = [c for c in numeric_cols if c != weight_col]
    else:
        numeric_cols_wo_w = numeric_cols
    summaries = {}
    summaries["missing"] = missing_summary(df)

    rows = []
    for col in numeric_cols_wo_w:
        x = pd.to_numeric(df[col], errors="coerce").values
        mask = np.isfinite(x); x = x[mask]
        if len(x) == 0:
            rows.append((col, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)); continue
        mu_unw = float(np.mean(x))
        se_unw = float(np.std(x, ddof=0) / max(1, np.sqrt(len(x)))) if len(x) > 0 else np.nan
        moe_unw = margin_of_error_95(se_unw)
        mu_w = np.nan
        if weight_col and weight_col in df.columns:
            w = pd.to_numeric(df.loc[mask, weight_col], errors="coerce").fillna(0).values
            mu_w, _ = weighted_mean_and_se(x, w)
        rows.append((col, len(x), float(np.nanmin(x)), float(np.nanmax(x)), mu_unw, se_unw, moe_unw, mu_w))
    summaries["numeric"] = pd.DataFrame(rows, columns=["column","n","min","max","mean_unw","se_unw","moe95_unw","mean_w"])

    cat_tables = {}
    for col in cat_cols:
        vc = df[col].astype(str).replace({"nan": np.nan}).value_counts(dropna=False)
        table = pd.DataFrame({"value": vc.index.astype(str), "count": vc.values})
        if weight_col and weight_col in df.columns:
            w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
            g = df.assign(_w=w).groupby(df[col].astype(str), dropna=False)["_w"].sum().reset_index()
            g.columns = ["value", "weighted_count"]
            total_w = g["weighted_count"].sum()
            g["weighted_prop"] = g["weighted_count"] / total_w if total_w > 0 else np.nan
            table = table.merge(g, on="value", how="left")
        cat_tables[col] = table
    summaries["categorical"] = cat_tables
    return summaries

# ---------- charts ----------
def fig_to_png_bytes(fig) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight")
    plt.close(fig)
    bio.seek(0)
    return bio.getvalue()

def plot_missing_bar(miss_df: pd.DataFrame, title: str) -> bytes:
    fig = plt.figure()
    vals = miss_df["missing"].values
    labels = miss_df.index.tolist()
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.title(title)
    plt.xticks(x, labels, rotation=75, ha="right")
    plt.ylabel("Missing Count")
    return fig_to_png_bytes(fig)

def plot_hist(series: pd.Series, title: str) -> bytes:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    fig = plt.figure()
    plt.hist(s, bins=30)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Frequency")
    return fig_to_png_bytes(fig)

def plot_box(series: pd.Series, title: str) -> bytes:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    fig = plt.figure()
    plt.boxplot(s, vert=True)
    plt.title(title)
    return fig_to_png_bytes(fig)

def plot_corr_heatmap(df: pd.DataFrame, title: str) -> bytes:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        fig = plt.figure(); plt.text(0.1, 0.5, "No numeric columns"); return fig_to_png_bytes(fig)
    corr = num.corr().fillna(0)
    fig = plt.figure()
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    return fig_to_png_bytes(fig)

def plot_topk_bar_counts(df: pd.DataFrame, col: str, k: int = 10) -> bytes:
    vc = df[col].astype(str).replace({"nan": np.nan}).value_counts().head(k)
    fig = plt.figure()
    plt.bar(np.arange(len(vc.index)), vc.values)
    plt.title(f"{col}: Top {k}")
    plt.xticks(np.arange(len(vc.index)), vc.index, rotation=75, ha="right")
    plt.ylabel("Count")
    return fig_to_png_bytes(fig)

# ---------- PDF ----------
def build_pdf_report(df0: pd.DataFrame, df_imp: pd.DataFrame, df_final: pd.DataFrame,
                     summaries: Dict[str, Any], weight_col: Optional[str],
                     outlier_flags: Optional[pd.DataFrame],
                     charts: Dict[str, bytes]) -> bytes:
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>CleanSheet v3 Report</b>", styles["Title"]))
    story.append(Paragraph(f"Weight column: <b>{weight_col or '(none)'}</b>", styles["Normal"]))
    story.append(Spacer(1, 8))

    rows = [
        ["Stage", "Rows", "Cols", "Memory (MB)"],
        ["Original", len(df0), df0.shape[1], f"{memory_usage_mb(df0):.3f}"],
        ["After Imputation", len(df_imp), df_imp.shape[1], f"{memory_usage_mb(df_imp):.3f}"],
        ["Final (after outliers/validation)", len(df_final), df_final.shape[1], f"{memory_usage_mb(df_final):.3f}"],
    ]
    table = Table(rows, hAlign="LEFT")
    table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    story.append(table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Missingness by Column (final)</b>", styles["Heading2"]))
    miss = summaries["missing"].reset_index().rename(columns={"index":"column"})
    miss_rows = [["Column","Missing","Missing%","Unique"]] + miss.head(30).values.tolist()
    table = Table(miss_rows, hAlign="LEFT", colWidths=[6*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    story.append(table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Numeric Summary</b>", styles["Heading2"]))
    num = summaries["numeric"]
    num_rows = [["Column","n","min","max","mean (unw)","SE (unw)","MOE95 (unw)","mean (w)"]]
    for _, r in num.iterrows():
        num_rows.append([r["column"], int(r["n"]) if not pd.isna(r["n"]) else "", r["min"], r["max"], r["mean_unw"], r["se_unw"], r["moe95_unw"], r["mean_w"]])
    table = Table(num_rows, hAlign="LEFT")
    table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    story.append(table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Categorical Snapshots (Top 10)</b>", styles["Heading2"]))
    for col, tab in list(summaries["categorical"].items())[:6]:
        story.append(Paragraph(f"<b>{col}</b>", styles["Heading3"]))
        view = tab.head(10).copy()
        headers = ["Value","Count"] + (["Weighted Count","Weighted Prop"] if "weighted_count" in view.columns else [])
        data = [headers]
        for _, r in view.iterrows():
            row = [str(r["value"]), int(r["count"])]
            if "weighted_count" in view.columns:
                row += [float(r.get("weighted_count", np.nan)), float(r.get("weighted_prop", np.nan))]
            data.append(row)
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
        story.append(t)
        story.append(Spacer(1, 6))

    def add_img(key, caption):
        if key in charts:
            img = RLImage(io.BytesIO(charts[key]), width=16*cm, height=9*cm)
            story.append(img); story.append(Paragraph(caption, styles["Italic"])); story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Visual Diagnostics</b>", styles["Heading2"]))
    add_img("miss_before", "Missingness by column — BEFORE imputation")
    add_img("miss_after", "Missingness by column — AFTER imputation")
    add_img("corr_final", "Correlation heatmap (final cleaned numeric columns)")
    for i in range(3):
        add_img(f"hist_before_{i}", "Histogram BEFORE outlier handling")
        add_img(f"hist_after_{i}", "Histogram AFTER outlier handling")
        add_img(f"box_before_{i}", "Boxplot BEFORE outlier handling")
        add_img(f"box_after_{i}", "Boxplot AFTER outlier handling")

    if outlier_flags is not None and outlier_flags.shape[1] > 0:
        story.append(Paragraph("<b>Outlier Flags</b>", styles["Heading2"]))
        counts = outlier_flags.sum(numeric_only=True)
        data = [["Column","Flagged Count"]] + [[c, int(v)] for c, v in counts.items()]
        t = Table(data, hAlign="LEFT"); t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
        story.append(t)

    doc.build(story)
    bio.seek(0)
    return bio.getvalue()

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE); st.caption(APP_DESC)

with st.expander("Quick Help (click me)", expanded=False):
    st.markdown("""
**Imputation** fills blanks. Numeric → median (safe). Category → most_frequent or constant ("Unknown").  
**Outliers**: flag/remove/clip extremes (Winsorize = clip by percentiles).  
**Validation**: range, allowed values, regex (email), skip-patterns.  
**Weights**: choose a weight column if some rows should count more/less.  
**Report**: download HTML/PDF with tables + charts.
    """)

st.sidebar.header("Upload")
file = st.sidebar.file_uploader("CSV/Excel", type=["csv","xlsx","xls"], help="Upload your dataset. Excel: choose a sheet next.")
cfg_file = st.sidebar.file_uploader("Optional: JSON Config", type=["json"], help="Use advanced control (schema, per-column impute/outliers, validation rules).")

st.sidebar.subheader("Defaults")
st.sidebar.number_input("KNN k", min_value=2, max_value=50, value=5, step=1, key="knn_k", help="Used if any numeric column uses KNN imputation.")
st.sidebar.number_input("Z-score threshold", min_value=1.0, max_value=10.0, value=3.0, step=0.5, key="z_thresh", help="Outlier if |z| > threshold.")
st.sidebar.number_input("IQR k", min_value=0.1, max_value=5.0, value=1.5, step=0.1, key="iqr_k", help="Outlier if outside [Q1 - k*IQR, Q3 + k*IQR].")
st.sidebar.slider("Winsorize quantiles", 0.0, 0.2, (0.01, 0.99), key="winsor_qs", help="Clip numeric values to [lower_q, upper_q] percentiles.")

st.sidebar.markdown("---")
st.sidebar.subheader("Inline Config (JSON)")
config_editor_text = st.sidebar.text_area("Paste JSON config (optional)", value="", height=160)

pipeline_config = {}
if cfg_file is not None:
    try:
        pipeline_config = json.load(cfg_file); st.success("Config loaded from file.")
    except Exception as e:
        st.error(f"Config JSON parse error: {e}")
elif config_editor_text.strip():
    try:
        pipeline_config = json.loads(config_editor_text.strip()); st.info("Using config from inline editor.")
    except Exception as e:
        st.error(f"Config JSON parse error: {e}")

pc = PipelineConfig(
    weight_column=pipeline_config.get("weight_column"),
    columns=pipeline_config.get("columns", {}),
    validations=[ValidationRule(**r) for r in pipeline_config.get("validations", [])]
)

if file is not None:
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            xls = pd.ExcelFile(file)
            sheet = st.selectbox("Select sheet", xls.sheet_names, help="Pick the sheet to import.")
            df = pd.read_excel(xls, sheet_name=sheet)
        st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns (memory ~ {memory_usage_mb(df):.3f} MB).")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        df = None
else:
    df = None

if df is not None:
    df0 = df.copy()

    st.header("Schema Mapping")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Detected dtypes**")
        st.dataframe(pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]}))
    with col2:
        st.markdown("**Rename Columns (optional)**")
        new_names = {}
        for c in df.columns:
            new = st.text_input(f"Rename '{c}'", value=(pc.columns.get(c, {}) or {}).get("rename_to", c), help="Type a new name or leave as-is.")
            if new != c: new_names[c] = new
        if st.button("Apply Renames", help="Click to rename columns"):
            df = df.rename(columns=new_names)
            new_columns_cfg = {}
            for old, conf in (pc.columns or {}).items():
                new_key = new_names.get(old, old); new_columns_cfg[new_key] = conf
            pc.columns = new_columns_cfg
            st.success("Renamed columns.")

    st.subheader("Weight Column")
    weight_col = st.selectbox("Weight column (optional)", ["(none)"] + df.columns.tolist(),
                              index=(0 if not pc.weight_column or pc.weight_column not in df.columns else df.columns.tolist().index(pc.weight_column)+1),
                              help="Choose your weight field if rows should count more/less. Leave as (none) if you don't have one.")
    if weight_col == "(none)": weight_col = None

    st.subheader("Set Column Types (optional)")
    type_map_opts = {"(auto)": None, "numeric": "numeric", "categorical": "categorical", "date": "date", "text": "text"}
    dtype_choices = {}
    for c in df.columns:
        default = (pc.columns or {}).get(c, {}).get("dtype", "(auto)")
        if default not in ["numeric","categorical","date","text"]: default = "(auto)"
        dtype_choices[c] = st.selectbox(f"{c}", list(type_map_opts.keys()), index=list(type_map_opts.keys()).index(default), key=f"dtype_{c}",
                                        help="Force a column to a specific type. Coercing text→number turns bad values into NaN.")
    if st.button("Apply Type Overrides", help="Convert columns to the selected types"):
        for c, choice in dtype_choices.items():
            t = type_map_opts[choice]
            if t == "numeric": df[c] = pd.to_numeric(df[c], errors="coerce")
            elif t == "date": df[c] = pd.to_datetime(df[c], errors="coerce")
            elif t == "categorical": df[c] = df[c].astype("category")
            elif t == "text": df[c] = df[c].astype(str)
        st.success("Applied dtype overrides.")

    st.markdown("---")
    st.header("Cleaning & Validation")
    logs: List[str] = []

    st.subheader("Missing-Value Imputation", help="Fill blanks. Numeric → median (safe). Category → most_frequent or constant.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    with st.expander("Configure per-column imputation (UI)", expanded=False):
        for c in df.columns:
            is_num = c in numeric_cols
            existing = (pc.columns or {}).get(c, {})
            existing_imp = (existing.get("impute") or {}).get("strategy", "(none)")
            choices = ["(none)", "mean", "median", "most_frequent", "constant", "knn"] if is_num else ["(none)", "most_frequent", "constant"]
            sel = st.selectbox(f"{c} strategy", choices, index=choices.index(existing_imp) if existing_imp in choices else 0, key=f"imp_{c}_strategy",
                               help="Pick how to fill NaNs for this column.")
            const_default = (existing.get("impute") or {}).get("constant", 0 if is_num else "")
            if sel == "constant":
                const_val = st.number_input(f"{c} constant value", value=float(const_default) if is_num and isinstance(const_default,(int,float)) else 0.0, key=f"imp_{c}_const_num") if is_num else st.text_input(f"{c} constant value", value=str(const_default), key=f"imp_{c}_const_text")
            else:
                const_val = None
            if sel != "(none)":
                conf = dict(existing); conf["impute"] = {"strategy": sel}
                if sel == "constant": conf["impute"]["constant"] = const_val
                if pc.columns is None: pc.columns = {}
                pc.columns[c] = conf
            else:
                if isinstance(existing, dict) and "impute" in existing:
                    existing2 = dict(existing); existing2.pop("impute", None); pc.columns[c] = existing2

    with st.expander("Defaults (apply when no column strategy is set)", expanded=False):
        st.selectbox("Default numeric imputation", ["(none)", "mean", "median", "most_frequent", "constant"], index=2, key="default_numeric_impute",
                     help="Used for numeric columns that still have NaNs and don't have a per-column choice.")
        st.number_input("Default constant (numeric)", value=0.0, key="default_constant_numeric")
        st.selectbox("Default categorical/text imputation", ["(none)", "most_frequent", "constant"], index=1, key="default_categorical_impute",
                     help="Used for non-numeric columns that still have NaNs and don't have a per-column choice.")
        st.text_input("Default constant (categorical/text)", value="", key="default_constant_categorical")

    run_imp = st.checkbox("Run Imputation", value=True, help="Execute the chosen imputation strategies.")
    if run_imp:
        df_imp = impute_missing(df, pc, logs)
    else:
        df_imp = df.copy()

    miss_before = missing_summary(df)
    miss_after = missing_summary(df_imp)
    charts = {}
    try:
        charts["miss_before"] = plot_missing_bar(miss_before, "Missingness — BEFORE imputation")
        charts["miss_after"] = plot_missing_bar(miss_after, "Missingness — AFTER imputation")
    except Exception as e:
        logs.append(f"Missingness chart failed: {e}")

    with st.expander("After imputation: handle any remaining nulls", expanded=False):
        drop_nulls = st.checkbox("Drop rows with any remaining nulls", value=False, key="drop_rows_with_nulls",
                                 help="If checked, any row that still contains a NaN after imputation will be removed.")
        if drop_nulls:
            before = len(df_imp); df_imp = df_imp.dropna(); logs.append(f"Dropped {before - len(df_imp)} rows with remaining nulls.")

    st.subheader("Outlier Detection & Handling", help="Find extreme values then flag/remove/clip them. Winsorize = clip by percentiles.")
    run_out = st.checkbox("Run Outlier Handling", value=False, help="Enable to apply per-column outlier settings from config.")
    if run_out:
        df_out, outlier_flags = handle_outliers(df_imp, pc, logs)
    else:
        df_out, outlier_flags = df_imp.copy(), pd.DataFrame(index=df_imp.index)

    num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    for i, col in enumerate(num_cols[:3]):
        try:
            charts[f"hist_before_{i}"] = plot_hist(df_imp[col], f"{col} — BEFORE outlier handling (hist)")
            charts[f"hist_after_{i}"]  = plot_hist(df_out[col], f"{col} — AFTER outlier handling (hist)")
            charts[f"box_before_{i}"]  = plot_box(df_imp[col], f"{col} — BEFORE outlier handling (box)")
            charts[f"box_after_{i}"]   = plot_box(df_out[col], f"{col} — AFTER outlier handling (box)")
        except Exception as e:
            logs.append(f"Chart failed for {col}: {e}")

    st.subheader("Rule-Based Validation", help="Add rules: range, allowed_values, regex, cross_field, skip_pattern. Optionally drop invalid rows.")
    run_val = st.checkbox("Run Validation", value=False)
    if run_val:
        vdf = validate_rules(df_out, pc.validations)
        st.write(f"Violations found: {len(vdf)}"); st.dataframe(vdf.head(200))
        drop_invalid = st.checkbox("Drop invalid rows", value=False, help="Remove rows that fail any rule.")
        if drop_invalid and len(vdf) > 0:
            before = len(df_out); df_out = df_out.drop(index=vdf["row_index"].unique(), errors="ignore"); logs.append(f"Dropped {before - len(df_out)} invalid rows.")

    st.header("Summaries & Diagnostics")
    st.subheader("Dataset Summary")
    ds = {
        "Rows": len(df_out),
        "Columns": df_out.shape[1],
        "Memory (MB)": round(memory_usage_mb(df_out),3),
        "Duplicate rows": int(df_out.duplicated().sum())
    }
    st.json(ds)

    st.subheader("Missingness (final)"); st.dataframe(missing_summary(df_out))

    try:
        charts["corr_final"] = plot_corr_heatmap(df_out, "Correlation heatmap (final numeric)")
        st.image(charts["corr_final"], caption="Correlation heatmap (final numeric)", use_column_width=True)
    except Exception as e:
        pass

    st.subheader("Categorical Distributions — Top 10")
    cat_cols = df_out.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols[:6]:
        try:
            img = plot_topk_bar_counts(df_out, col, 10)
            st.image(img, caption=f"{col}: Top 10 values", use_column_width=True)
        except Exception as e:
            pass

    summaries = compute_summaries(df_out, weight_col)
    st.subheader("Numeric Summary"); st.dataframe(summaries["numeric"])
    st.subheader("Categorical Summary (per column)")
    for col, tab in summaries["categorical"].items():
        with st.expander(f"{col}", expanded=False):
            st.dataframe(tab)

    st.header("Report & Downloads")
    def render_html_report():
        numeric_html_rows = ""
        if "numeric" in summaries and not summaries["numeric"].empty:
            for _, r in summaries["numeric"].iterrows():
                numeric_html_rows += f"<tr><td>{r['column']}</td><td>{int(r['n']) if not pd.isna(r['n']) else ''}</td><td>{r['min']}</td><td>{r['max']}</td><td>{r['mean_unw']}</td><td>{r['se_unw']}</td><td>{r['moe95_unw']}</td><td>{r['mean_w']}</td></tr>"
        miss_html = missing_summary(df_out).reset_index().rename(columns={'index':'column'}).to_html(index=False)
        html = f"""
        <html><head><meta charset='utf-8'/><title>CleanSheet v3 Report</title>
        <style>body{{font-family:system-ui,Arial,sans-serif;padding:20px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:6px}}th{{background:#f7f7f7}}</style></head>
        <body>
          <h1>CleanSheet v3 Report</h1>
          <div>Rows {len(df0)}→{len(df_imp)}→{len(df_out)} | Cols {df0.shape[1]}→{df_imp.shape[1]}→{df_out.shape[1]} | Weight: {weight_col or '(none)'} | Memory (final): {memory_usage_mb(df_out):.3f} MB</div>
          <h2>Missingness (final)</h2>
          {miss_html}
          <h2>Numeric Summary</h2>
          <table><tr><th>Column</th><th>n</th><th>min</th><th>max</th><th>mean_unw</th><th>se_unw</th><th>moe95_unw</th><th>mean_w</th></tr>{numeric_html_rows}</table>
        </body></html>
        """
        return html

    html_report = render_html_report()
    st.download_button("Download Cleaned CSV", data=bytes_from_df_csv(df_out), file_name="cleaned.csv")
    st.download_button("Download Cleaned Excel", data=bytes_from_df_excel(df_out), file_name="cleaned.xlsx")
    st.download_button("Download Report (HTML)", data=html_report.encode("utf-8"), file_name="report.html")

    try:
        pdf_bytes = build_pdf_report(df0, df_imp, df_out, summaries, weight_col, outlier_flags, charts)
        st.download_button("Download Report (PDF)", data=pdf_bytes, file_name="report.pdf")
    except Exception as e:
        st.error(f"PDF build failed: {e}")
else:
    st.info("Upload a CSV/Excel file to begin.")

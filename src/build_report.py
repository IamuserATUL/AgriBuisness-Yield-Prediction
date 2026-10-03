"""Builds reports/Predictive_Modeling_Report.docx from outputs/results.json and outputs/figures/*.png.

All numbers and interpretive sentences come from results.json, so the report always matches the run.
Usage: python src/build_report.py
"""
import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

DATASET_URL = "https://www.kaggle.com/datasets/abhinand05/crop-production-in-india"


def fmt(n, d=0):
    return f"{n:,.{d}f}"


def table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = h
        for r in t.rows[0].cells[i].paragraphs[0].runs:
            r.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()


def figure(doc, figdir, name, caption, width=6.0):
    p = figdir / name
    if not p.exists():
        return
    doc.add_picture(str(p), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(caption)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.runs[0].italic = True
    c.runs[0].font.size = Pt(9)


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def build(R, figdir, out_path):
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    D, cl, tu = R["data"], R["cleaning"], R["tuning"]
    M = {m["model"]: m for m in R["models"]}
    base = M["Baseline (crop median)"]
    best_name = R["best_model"]
    best = M[best_name]
    noh = M["Gradient boosting (no history features)"]
    tuned = M["Gradient boosting (tuned)"]
    gain_mae = (1 - best["mae_tha"] / base["mae_tha"]) * 100
    gain_log = (1 - best["log_mae"] / base["log_mae"]) * 100
    hist_gain = (1 - tuned["log_mae"] / noh["log_mae"]) * 100

    doc.add_heading("Predictive Modeling for Agribusiness Insights: Crop Yield Prediction", 0)
    doc.add_paragraph("Week 4 Task: Development, tuning and evaluation of a predictive model for crop yield in India")

    if "SYNTHETIC" in R["input_file"].upper():
        w = doc.add_paragraph()
        r = w.add_run("WARNING: this report was generated from SYNTHETIC test data. None of the results below are real. "
                      "Re-run the pipeline on the real dataset.")
        r.bold = True
        r.font.color.rgb = RGBColor(0xB2, 0x3A, 0x48)

    # 1 Introduction
    doc.add_heading("1. Introduction and objectives", 1)
    doc.add_paragraph(
        "Agribusiness firms, lenders, insurers and policy makers all need a view of how much a crop is likely to yield "
        "before the harvest is in. This project simulates how data science is applied to that problem: it builds a "
        "predictive model that estimates the yield (tonnes per hectare) of a crop in a district, using historical "
        "records, and evaluates how well it would have performed on years it has never seen.")
    doc.add_paragraph("Objectives:")
    bullets(doc, [
        "Frame yield estimation as a supervised regression problem and justify the modeling approach.",
        "Engineer features from location, crop, season, land area and each series' own history.",
        "Compare a simple baseline against linear, bagged-tree and boosted-tree models; tune the best family.",
        "Evaluate honestly with a time-based hold-out, several metrics and an ablation of the history features.",
        "Interpret the model outputs for agribusiness decisions and discuss challenges and improvements.",
    ])

    # 2 Problem and data
    doc.add_heading("2. Problem formulation and data", 1)
    doc.add_paragraph(
        f"Target: crop yield = Production / Area (tonnes per hectare), modeled as log(1 + yield) because yields are "
        f"strongly right-skewed and differ by orders of magnitude between crops (for example sugarcane versus pulses). "
        f"Predictions are converted back to tonnes per hectare for reporting.")
    doc.add_paragraph(
        f"Data: 'Crop Production in India' (data.gov.in, compiled on Kaggle: {DATASET_URL}); file used: {R['input_file']}. "
        f"This is the same public dataset explored in Week 3. After cleaning and keeping fully reported years, the "
        f"modeling table has {fmt(D['rows_model'])} district-level records covering {D['crops']} crops and {D['states']} states/UTs.")

    # 3 Assumptions
    doc.add_heading("3. Assumptions", 1)
    bullets(doc, [
        "Reported Area and Production are accurate enough that Production / Area is a valid yield; extreme values were "
        "treated as errors (see preprocessing).",
        "Coconut is excluded because it is recorded in nuts while all other crops are in tonnes.",
        "The scenario is a one-year-ahead forecast: when predicting year t, the model may use the observed yield of "
        "the same district, crop and season in earlier years, but never year t itself.",
        "Cultivated area for the target year is known at prediction time (sowing area is typically reported or "
        "estimated before harvest).",
        "Weather, irrigation, prices and input use are not in the dataset, so their effect appears only indirectly "
        "through location, season and past yield.",
        "Patterns in earlier years are assumed to remain broadly valid in the following years (no structural break).",
    ])

    # 4 Preprocessing
    doc.add_heading("4. Preprocessing steps", 1)
    bullets(doc, [
        f"Started from {fmt(cl['rows_raw'])} raw rows. Text fields were trimmed, exact duplicates and rows with missing "
        f"values or zero area were removed, leaving {fmt(cl['rows_after_basic_cleaning'])} rows.",
        f"Extreme yield outliers (beyond 3 x IQR of log-yield within each crop) were removed: "
        f"{fmt(cl['extreme_outliers_removed'])} rows. These are almost certainly unit or data-entry errors and would "
        "dominate squared-error losses.",
        ("Years with far fewer records than usual look incompletely reported and are excluded: "
         + (", ".join(map(str, D["dropped_partial_years"])) + "." if D["dropped_partial_years"]
            else "none were found in this data.")),
        "Categorical variables were one-hot encoded for the linear model and ordinal-encoded for tree models; "
        "unseen categories at prediction time are handled safely.",
        "Missing history values (first year of a series) are imputed with the median plus a missing-indicator for "
        "linear and random-forest models; gradient boosting handles missing values natively.",
        "Numeric inputs were standardised for the linear model only (trees do not need scaling).",
    ])

    # 5 Feature engineering and selection
    doc.add_heading("5. Feature engineering and feature selection", 1)
    table(doc, ["Feature", "Type", "Reason"], [
        ["Crop", "Categorical", "Crop is the strongest single determinant of yield level"],
        ["State_Name", "Categorical", "Captures regional agro-climate, soil and farming practice"],
        ["Season", "Categorical", "Kharif / Rabi / Summer cropping conditions differ"],
        ["Crop_Year", "Numeric", "Captures slow technology and productivity trends"],
        ["log_area", "Numeric (engineered)", "Scale of cultivation; log tames the heavy skew"],
        ["yield_lag1", "Numeric (engineered)", "Same district-crop-season yield in the previous year"],
        ["yield_prior_mean", "Numeric (engineered)", "Mean log-yield of all earlier years for that series"],
    ])
    doc.add_paragraph(
        "Leakage control: the two history features are built only from years strictly before the row's year, and the "
        "train/test split is by time, so no information from the test period is used in training.")
    doc.add_paragraph(
        "Feature selection was done by domain reasoning (a deliberately small feature set) and verified afterwards "
        "with permutation importance on the held-out years. District name was not used as a feature because it has "
        "hundreds of levels and the district's own history already carries that information.")

    # 6 Modeling approach and rationale
    doc.add_heading("6. Modeling approach and rationale", 1)
    doc.add_paragraph("Four model families were compared against a simple baseline, ordered from simple to flexible:")
    table(doc, ["Model", "Why it was included"], [
        ["Baseline: crop median", "Minimum bar. Any useful model must beat 'predict the typical yield of this crop'"],
        ["Ridge regression", "Interpretable linear benchmark with regularisation against collinearity"],
        ["Random forest", "Bagged trees: captures non-linearity and interactions, robust to noise"],
        ["Gradient boosting (HistGradientBoosting)", "Usually the strongest on tabular data; native categorical and missing-value support; fast"],
    ])
    doc.add_paragraph(
        "Why regression on log-yield: the outcome is a continuous quantity, and modeling the log makes errors relative "
        "rather than absolute, which is what matters when crops range from under 1 to over 50 tonnes per hectare.")

    # 7 Training, tuning, validation
    doc.add_heading("7. Model training, tuning and validation", 1)
    doc.add_paragraph(
        f"Validation design: a time-based hold-out. Models were trained on {D['train_years'][0]}-{D['train_years'][1]} "
        f"({fmt(D['train_rows'])} records) and tested on {D['test_years'][0]}-{D['test_years'][-1]} "
        f"({fmt(D['test_rows'])} records). A random split would mix years and overstate accuracy, because the "
        f"future would leak into training. History features were available for {D['history_coverage_test_pct']}% of test records.")
    doc.add_paragraph(
        f"Hyper-parameter tuning: randomized search ({tu['n_iter']} candidates, {tu['cv_folds']}-fold cross-validation, "
        f"{fmt(tu['tuning_rows'])} training rows, scored by mean absolute error on the log scale). Selected parameters:")
    table(doc, ["Parameter", "Value"], [[k, f"{v:.4g}" if isinstance(v, float) else v] for k, v in tu["best_params"].items()])
    doc.add_paragraph(
        f"Cross-validated R-squared of the tuned model on the training subsample was {tu['cv_r2_mean']} "
        f"(standard deviation {tu['cv_r2_std']} across folds), indicating stable performance.")

    # 8 Metrics
    doc.add_heading("8. Performance metrics and evaluation criteria", 1)
    table(doc, ["Metric", "What it tells us"], [
        ["MAE (t/ha)", "Average size of the error in the crop's own units; easy to explain to farmers and buyers"],
        ["RMSE (t/ha)", "Like MAE but penalises large misses more; dominated by high-yield crops such as sugarcane"],
        ["R-squared (log scale)", "Share of variation in log-yield explained. Inflated by large differences between crops, so read with the improvement over baseline"],
        ["Median absolute % error", "Typical relative error of one prediction: the most intuitive accuracy measure here"],
        ["Share within +/-25%", "Share of predictions that land within a quarter of the true yield (practical usability)"],
        ["Improvement over baseline", "How much lower the error is than the crop-median baseline: the real measure of added skill"],
    ])
    doc.add_paragraph(
        "Success criteria used: (1) the model must beat the baseline on the unseen test years; (2) train and test "
        "performance must be close (no overfitting); (3) errors should be unbiased (residual mean near zero); and "
        "(4) performance should be reasonable across the major crops, not only on average.")

    # 9 Results
    doc.add_heading("9. Results", 1)
    rows = [[m["model"], f"{m['mae_tha']:.2f}", f"{m['rmse_tha']:.2f}", f"{m['log_r2']:.3f}", f"{m['mdape_pct']:.1f}%",
             f"{m['within25_pct']:.0f}%"] for m in R["models"]]
    table(doc, ["Model", "MAE (t/ha)", "RMSE (t/ha)", "R2 (log)", "Median % error", "Within 25%"], rows)
    figure(doc, figdir, "01_model_comparison.png", "Figure 1: Model comparison on the held-out test years.", 6.3)
    doc.add_paragraph(
        f"Interpretation. The best model is {best_name}. It reaches a median error of {best['mdape_pct']:.1f}% and "
        f"{best['within25_pct']:.0f}% of predictions fall within 25% of the actual yield. Its MAE is "
        f"{best['mae_tha']:.2f} t/ha versus {base['mae_tha']:.2f} t/ha for the baseline, an error reduction of "
        f"{gain_mae:.0f}% in t/ha and {gain_log:.0f}% on the log scale. "
        + ("" if best["log_r2"] < 0.9 else
           "The high R-squared mainly reflects that crops differ enormously in yield level; the improvement over "
           "baseline is the fairer indicator of skill. "))
    ranks = sorted(R["models"][:5], key=lambda m: m["log_mae"])
    doc.add_paragraph(
        f"Ranked by log-scale MAE: " + " < ".join(f"{m['model']} ({m['log_mae']:.3f})" for m in ranks) +
        ". Tree-based models can pick up non-linear effects and interactions between crop, state and season that a "
        "linear model cannot, which is why they are expected to lead; differences among the top models are small "
        "enough that simplicity and speed can legitimately drive the final choice.")
    doc.add_heading("9.1 Value of history features (ablation)", 2)
    doc.add_paragraph(
        f"Removing the two history features from the tuned model raises its log-scale MAE from {tuned['log_mae']:.3f} to "
        f"{noh['log_mae']:.3f} ({hist_gain:.0f}% worse) and its median error from {tuned['mdape_pct']:.1f}% to "
        f"{noh['mdape_pct']:.1f}%. " +
        ("The past performance of a district-crop series is therefore a valuable predictor, which supports the "
         "design of the feature set." if hist_gain > 2 else
         "The history features add little here, so a simpler feature set would do almost as well."))
    doc.add_heading("9.2 Fit quality and residuals", 2)
    figure(doc, figdir, "02_actual_vs_predicted.png", "Figure 2: Actual versus predicted log-yield on test years.", 4.6)
    figure(doc, figdir, "03_residuals.png", "Figure 3: Residual distribution and residuals against predictions.", 6.2)
    tv = R["train_vs_test"]
    rs = R["residuals"]
    gap = tv["train_log_r2"] - tv["test_log_r2"]
    doc.add_paragraph(
        f"Interpretation. Points hugging the diagonal in Figure 2 indicate good agreement. The mean residual is "
        f"{rs['mean']:+.3f} on the log scale (spread {rs['std']:.3f}); " +
        ("a near-zero mean means the model is not systematically over- or under-predicting. " if abs(rs["mean"]) < 0.05 else
         "a non-zero mean suggests a systematic bias that should be investigated, for example a shift in yields between the training and test years. ") +
        f"Train R-squared is {tv['train_log_r2']:.3f} against {tv['test_log_r2']:.3f} on test (gap {gap:.3f}), " +
        ("so there is no meaningful overfitting." if gap < 0.03 else "which signals some overfitting; stronger regularisation or simpler trees would help."))

    # 10 Interpretation for agribusiness
    doc.add_heading("10. Interpretation of model outputs for agribusiness", 1)
    if "importance" in R:
        imp = R["importance"]
        figure(doc, figdir, "04_feature_importance.png", "Figure 4: Permutation importance of each feature on test data.", 5.6)
        top = [i for i in imp if i["importance"] > 0][:3]
        doc.add_paragraph(
            "Drivers of yield. Permutation importance measures how much the error grows when a feature is shuffled. "
            + "The most influential features are " + ", ".join(f"{i['feature']} ({i['importance']:.3f})" for i in top)
            + ". Features near zero add little once the others are known. These are associations learned from data, not "
            "causal effects.")
    ce = R.get("crop_errors", [])
    if ce:
        figure(doc, figdir, "05_error_by_crop.png", "Figure 5: Median absolute percentage error by crop (test years).", 5.6)
        ce_s = sorted(ce, key=lambda c: c["mdape_pct"])
        doc.add_paragraph(
            f"Reliability by crop. The most predictable major crop is {ce_s[0]['crop']} (median error "
            f"{ce_s[0]['mdape_pct']:.1f}%) and the least predictable is {ce_s[-1]['crop']} "
            f"({ce_s[-1]['mdape_pct']:.1f}%). Predictions for high-error crops should carry wider safety margins.")
    doc.add_paragraph("How the model could be used in practice:")
    bullets(doc, [
        "Procurement and logistics: turn predicted yield x planned area into expected tonnage to plan storage, "
        "transport and processing capacity by district.",
        "Credit and insurance: flag district-crop combinations whose predicted yield is far below their history as "
        "higher risk, and size loans or premiums accordingly.",
        "Input suppliers: target seed and fertiliser outreach to areas where predicted yield lags the state benchmark.",
        "Planning with uncertainty: use the median error as a planning band (for example +/- the typical percentage "
        "error) rather than treating a prediction as exact.",
    ])

    # 11 Challenges
    doc.add_heading("11. Challenges encountered and strategies for improvement", 1)
    table(doc, ["Challenge", "How it was handled / how to improve"], [
        ["Highly skewed target and large differences between crops", "Log-transform the target; report relative errors; evaluate per crop"],
        ["Outliers and likely data-entry errors", f"Removed {fmt(cl['extreme_outliers_removed'])} extreme rows with a per-crop IQR rule; next: validate against source reports"],
        ["Risk of leakage through history features and random splits", "History uses only earlier years; time-based hold-out"],
        ["No weather, irrigation, price or input data", "Largest opportunity: add rainfall, temperature, irrigated-area share and fertiliser use"],
        ["Missing history for new district-crop series", "Imputation with indicators / native handling; fall back to state-crop averages"],
        ["Overfitting risk of flexible models", "Cross-validated tuning, regularisation, train-vs-test comparison"],
        ["Different crops have different error profiles", "Fit crop-specific models or add prediction intervals (quantile regression)"],
    ])
    doc.add_paragraph("Further improvement ideas: wider hyper-parameter search; ensembling several models; "
                      "quantile or conformal prediction intervals; rolling-origin back-testing across many years; "
                      "neighbouring-district features; and SHAP values for richer explanations.")

    # 12 Conclusion
    doc.add_heading("12. Conclusion", 1)
    doc.add_paragraph(
        f"A {best_name.lower()} model predicts district-level crop yield for unseen years with a median error of "
        f"{best['mdape_pct']:.1f}%, cutting error by about {gain_mae:.0f}% relative to a crop-median baseline. "
        f"Past yield of the same series is a useful input (removing it makes the log-scale error {hist_gain:.0f}% worse), "
        "and the evaluation shows how reliable the model is overall and by crop. The main limitation is the "
        "absence of weather and input data; adding them is the clearest path to better accuracy.")

    # 13 Tools
    doc.add_heading("13. Tools and reproducibility", 1)
    table(doc, ["Tool", "Purpose"], [
        ["Python 3, pandas, NumPy", "Data cleaning and feature engineering"],
        ["scikit-learn", "Pipelines, Ridge, Random Forest, HistGradientBoosting, search, metrics, permutation importance"],
        ["Matplotlib, Seaborn", "Charts"],
        ["python-docx", "Automatic generation of this report"],
        ["Git / GitHub", "Version control; code and instructions in the repository"],
    ])
    doc.add_paragraph("Reproduce: python src/model.py, then python src/build_report.py (random seed fixed at 42).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results.json")
    ap.add_argument("--figures", default="outputs/figures")
    ap.add_argument("--out", default="reports/Predictive_Modeling_Report.docx")
    a = ap.parse_args()
    build(json.load(open(a.results)), Path(a.figures), Path(a.out))
    print(f"Report written to {a.out}")

"""Generates a SYNTHETIC file with the same columns as the real dataset.

*** FOR SMOKE-TESTING THE PIPELINE ONLY. Never report results from this file. ***
The output name contains 'SYNTHETIC' so build_report.py stamps a warning on any report made from it.

Run from the project root:  python tests/make_synthetic_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
states = ["Uttar Pradesh", "Maharashtra", "Punjab", "Rajasthan", "Karnataka", "Tamil Nadu", "Bihar", "Gujarat"]
crops = {"Rice": 2.4, "Wheat": 2.8, "Sugarcane": 60, "Maize": 2.5, "Bajra": 1.2,
         "Groundnut": 1.1, "Cotton(lint)": 0.4, "Coconut": 7000}
season_of = {c: rng.choice(["Kharif", "Rabi", "Whole Year", "Summer"]) for c in crops}
rows = []
for s in states:
    s_eff = float(np.exp(rng.normal(0, 0.25)))
    for d in range(10):
        d_eff = float(np.exp(rng.normal(0, 0.2)))
        for c, base in crops.items():
            if rng.random() < 0.7:
                area0 = float(np.exp(rng.normal(7, 1.2)))
                for y in range(1997, 2016):
                    if rng.random() < 0.9:
                        yld = base * s_eff * d_eff * (1 + 0.012 * (y - 1997)) * float(np.exp(rng.normal(0, 0.15)))
                        area = area0 * float(np.exp(rng.normal(0, 0.1)))
                        rows.append((s, f"{s[:3]}-D{d}", y, season_of[c] + "     ", c, area, area * yld))
pd.DataFrame(rows, columns=["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production"]
             ).to_csv("data/raw/SYNTHETIC_smoke_test.csv", index=False)
print("wrote data/raw/SYNTHETIC_smoke_test.csv")

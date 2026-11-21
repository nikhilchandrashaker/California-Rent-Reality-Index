# Orange County Rent Reality Index

A data-driven look at Orange County, CA rental prices: which areas are most expensive, what drives rent, and which listings look cheap or expensive relative to a model's expectation.

**Data:** 2,500 Zillow rental search results (one Orange County search), snapshot 2025-05-21 (assumed from the project brief; the raw file has no date column). These are asking rents, not signed leases.

> **Scope note:** the original brief targeted all of California, but the supplied file covers Orange County only (45 cities, 85 ZIP codes). Everything here is an Orange County index. It is one snapshot, so it says nothing about rent growth or time to lease.

## Start here

| File | What it is |
|---|---|
| `OC_Rent_Reality_Executive_Report.pdf` | 4-page executive report (also available as an editable doc) |
| Dashboard (published page) | Interactive dashboard; also saved as `oc_rent_dashboard.html` |
| `OC_Rent_Models_and_Value_Score.xlsx` | Models, Rent Value Scores, city premiums, rankings |

## All deliverables

| File | Contents |
|---|---|
| `OC_Rent_Clean_Data_and_Audit.xlsx` | Step 1: clean unit table, property table, data dictionary, quality audit |
| `oc_units_clean.csv` | 3,108 unit-level records with quality flags and `model_ready` |
| `oc_properties_master.csv` | 2,500 property-level records (one per Zillow result) |
| `OC_Rent_Models_and_Value_Score.xlsx` | Step 2: model comparison, Value_Scores, location vs features, city premiums, bedroom effect, rankings, assumptions |
| `oc_rent_value_scores.csv` | Value Score for every model-ready record |
| `OC_Rent_Geographic_Analysis.xlsx` | Step 3: city and ZIP rankings, top/bottom lists, price tiers |
| `maps/` | Six PNGs: median rent, premium, value score, price tiers, 2BR listings, major-markets chart |
| `oc_rent_dashboard.html` | Step 4: self-contained dashboard (6 pages) |
| `scripts/` | Python scripts that reproduce everything |

Workbook summary and metric cells are live Excel formulas over the data sheets, so they recalculate if you edit data or thresholds (blue text = hardcoded model output or editable input).

## Key definitions

- **Unit-level record:** one rent per row. Individual units come from `unformattedPrice`; building floor plans are exploded from `units/0-3` and parsed from text such as "$2,499+".
- **model_ready:** has beds and rent, and is not a hard error (square footage under 200 or over 10,000, or rent more than 3x or under 0.4x Zillow's own rent estimate). 3,082 of 3,108 records.
- **Rent Value Score:** `(predicted rent - actual rent) / predicted rent x 100`. Predictions are out-of-fold (5-fold cross-validation grouped by address), so no listing is scored by a model that saw its own building. Positive = cheaper than expected.
- **Premium vs same-size home:** median of `actual / features-only prediction` minus 1, where the features-only model uses beds, baths, sqft and type but no location.
- **Value labels** (editable on the Assumptions sheet): Strong value >= 15, Good >= 5, Fair > -5, Premium > -15, else Overpriced.

## Known limitations

1. Amenity flags (pool, AC, fireplace, spa) are FALSE for every individual unit in the source, so amenity and luxury premiums cannot be measured.
2. Building floor plans have no baths or sqft and use "starting at" prices, so they are modeled separately and lean slightly cheap.
3. Rents above $15,000 are kept but flagged. Ultra-luxury areas (Newport Coast, Sunset Beach) are under-predicted, so their negative scores mean "very expensive," not necessarily overpriced.
4. Missing coordinates (88 rows) were filled with city medians for modeling only.
5. Zillow's `rentZestimate` and `zestimate`, and the duplicate `price` fields, are excluded as predictors to avoid target leakage.
6. ZIP medians rest on 10-60 listings each; treat small ZIPs as indicative.
7. The Rent Value Score is relative pricing within this dataset, not verified quality or true market rent.

## Reproducing the analysis

Python 3.12 with `pandas`, `numpy`, `scikit-learn`, `openpyxl`, `matplotlib`, `reportlab`, `Pillow`. LibreOffice is used to recalculate the workbooks.

Scripts pass data between steps through pickle files in the working directory and use absolute paths (`/mnt/user-data/uploads/...` for input, `/mnt/user-data/outputs/...` for output). Edit those paths for your machine, then run from one working directory in this order:

1. `build_clean.py` - reads the raw CSV, builds `units.pkl` and `props.pkl`
2. `make_xlsx.py` - writes the clean data and audit workbook and CSVs (creates `units2.pkl`)
3. `models.py` - cross-validated models, value scores, importances, premiums
4. `make_models_xlsx.py` - writes the models workbook and value score CSV
5. `geo.py` - ZIP aggregation, clustering, maps
6. `make_geo_xlsx.py` - writes the geographic workbook
7. `dash.py` then substitute `dash_data.json` into `shell.html` at `__DATA__` - builds the dashboard
8. `make_pdf.py` - builds the executive report PDF

Random seed is 42 throughout.

## Suggested next steps

- Add more Zillow searches to cover the rest of California.
- Collect a second snapshot to measure rent change.
- Add amenity or description data to test luxury premiums.

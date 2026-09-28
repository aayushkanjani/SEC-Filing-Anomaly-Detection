# SEC Filing Anomaly Detection — Deliverable 1: Data Extraction

This deliverable builds a secondary dataset of financial-statement features from SEC EDGAR. The dataset is designed as input for statistical anomaly detection (Isolation Forest, Z-score) aimed at spotting patterns that can indicate financial-reporting fraud.

## Repository structure

```
├── README.md
├── requirements.txt
├── .gitignore                         # excludes edgar_cache/ (raw downloads)
├── notebooks/
│   └── 01_edgar_data_extraction.ipynb # extraction + feature engineering (run top to bottom)
├── data/
│   └── processed/                     # secondary dataset produced by the notebook
│       ├── quarterly_features.csv
│       ├── annual_features.csv
│       ├── facts_long.csv
│       └── coverage_report.csv
├── docs/
│   └── data_dictionary.csv            # every output column: type, unit, description, source/formula
└── tests/
    └── test_fixture_generator.py      # simulated EDGAR responses for offline testing
```

Running the notebook produces the dataset in `data/processed/`:

| Output CSV | Grain | Use |
|---|---|---|
| `quarterly_features.csv` | company × fiscal quarter | **Main secondary dataset** (≈90 columns) |
| `annual_features.csv` | company × fiscal year | Annual ratios and Beneish M-Score |
| `facts_long.csv` | one row per XBRL fact | Audit trail: every number traces to a filing accession number |
| `coverage_report.csv` | company | Completeness check per company |

## Quick start

1. Install the dependencies: `pip install pandas numpy requests jupyter`.
2. Set your SEC identity as an environment variable, for example `export SEC_USER_AGENT="Your Name you@domain.com"` (Windows PowerShell: `$env:SEC_USER_AGENT="Your Name you@domain.com"`). The SEC rejects anonymous automated requests. You can also edit `USER_AGENT` in the notebook directly, but then your e-mail ends up in the repository.
3. Optionally edit `CONTROL_TICKERS`, `REFERENCE_TICKERS`, `START_YEAR`.
4. Open `notebooks/01_edgar_data_extraction.ipynb` and run all cells. The default 26 companies take roughly 1–3 minutes on a first run. Raw JSON is cached in `edgar_cache/` at the repo root (git-ignored), so later runs work offline and finish in seconds.

## Data sources

All data comes from the SEC's free, keyless JSON APIs:

| Endpoint | Provides |
|---|---|
| `https://www.sec.gov/files/company_tickers.json` | Ticker → CIK mapping |
| `https://data.sec.gov/submissions/CIK##########.json` | Company name, SIC industry, fiscal-year-end |
| `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | Every XBRL fact the company has filed, with period dates, filing date, form type and accession number |

XBRL financial data became mandatory for large accelerated filers from 2009 and for all other US GAAP filers by 2011. The default `START_YEAR = 2014` therefore gives 10+ years for most companies, comfortably exceeding the 5-year requirement. The notebook requests at most about 8 per second, under the SEC's limit of 10.

## Methodology

**1. Tag standardisation.** Companies describe the same line item with different US-GAAP tags, and tags change over time. The largest such change was ASC 606 in 2018, when many firms moved from `SalesRevenueNet` to `RevenueFromContractWithCustomerExcludingAssessedTax`. Each standard item (23 in total) has an ordered list of candidate tags. For every period the highest-priority tag with a value is used, and the chosen revenue tag is stored. `revenue_tag_changed` marks quarters where the tag switched, so jumps caused purely by re-tagging can be excluded. Accounting identities fill gaps: gross profit = revenue − cost of revenue, and liabilities = total liabilities & equity − equity.

**2. Point-in-time values and restatement detection.** A single quarter is reported several times: once in its own 10-Q, then as a comparative in later filings. The dataset uses the value **as first reported** for every financial figure. This matters for two reasons:
- It is what investors actually saw at the time, which avoids look-ahead bias. This is the standard choice in fraud research, since the later restatement is the outcome being predicted.
- It keeps derived quarters consistent. Later filings recast earlier years after spin-offs and discontinued operations. Subtracting a recast full year from a 9-month figure that was never recast produces a meaningless Q4. An earlier version of this pipeline did this, which showed up as false anomalies for 3M, Johnson & Johnson and Honeywell.

The latest filed value is also kept. When it differs from the first reported value by more than 0.5%, the fact is marked as restated. These comparisons appear in `revenue_latest`, `revenue_restatement_pct`, `n_items_restated` and `n_bs_items_restated`. `any_amended` marks values that came from 10-K/A or 10-Q/A amendments.

**3. Discrete quarters.** Q4 is never filed on its own, and cash-flow items appear in 10-Qs only as year-to-date totals. Discrete quarters are built in this order of preference:
- `reported_3m`: a filed 3-month value.
- `ytd_diff`: the difference of consecutive year-to-date values with the same start date (Q2 = 6M − 3M, Q4 = 12M − 9M).
- `annual_minus_3q`: the full year minus the three known quarters.

The method used for revenue is stored in `revenue_method`. After quarters are built, each fiscal year is checked against its annual revenue:
- A quarter holding more than 90% of the year's revenue is a source-data error, where a full-year value was tagged as a quarter. Its revenue is set to missing and `dq_quarter_looks_annual` is set. This pattern appeared in Oracle's FY2018 data.
- A year whose four quarters differ from the annual figure by more than 2% sets `dq_quarter_sum_mismatch`.

**4. Fiscal calendar.** Fiscal year and quarter come from each company's fiscal-year-end. Dates are shifted by 14 days so that 52/53-week years (for example a year ending 2 January) and retailer calendars are labelled correctly. `fiscal_year` is always **the calendar year in which the fiscal year ends**. Some retailers use a different convention: Home Depot, Lowe's and Target name their fiscal year after the year it starts, so their own labels are one lower than this dataset's. Walmart already uses the ending year. `calendar_quarter` is provided for comparing companies with different fiscal years.

**5. Lag matching by date.** Previous-quarter values are matched to periods ending 75–125 days earlier, and prior-year values to periods ending 340–390 days earlier. Lags are not taken by row position, so a missing quarter yields a blank value instead of a silently wrong growth rate.

**6. Features.** Features are grouped into five families:
- **Growth:** revenue QoQ and YoY (YoY removes seasonality), plus YoY growth of COGS, SG&A, net income, receivables, inventory and deferred revenue.
- **Expense-to-revenue ratios and margins:** COGS, SG&A, R&D and operating expenses relative to revenue; total expense ratio = (revenue − operating income) / revenue; gross, operating and net margin; and year-over-year changes in these ratios.
- **Earnings-quality red flags:** days sales outstanding (DSO), days inventory outstanding (DIO), receivables growth minus revenue growth (a channel-stuffing signal), inventory growth minus revenue growth, Sloan accruals ((net income − operating cash flow) / average assets), and cash conversion.
- **Unusual jumps and drops:** trailing Z-scores (`z_*`) of eight key metrics, each compared with the company's own previous 8 quarters (minimum 4). Only past data is used, so there is no look-ahead bias. Robust median/MAD versions (`rz_*`) are included because a single extreme quarter inflates an ordinary standard deviation. `n_abs_z_gt_3` and `max_abs_z` summarise each quarter.
- **Beneish M-Score** (annual file): all eight indices (DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA), the combined score, and a flag at the conventional −1.78 threshold.

## Company universe

The universe has two groups. The **control group** has 20 large, long-listed companies. The **reference group** has 6 companies with publicly reported restatements or SEC accounting actions: KHC, UAA, SMCI, MDXG, PLUG and MAT. The reference group gives weak labels for sanity-checking the Deliverable 2 models. Each case should be confirmed against the SEC's Accounting and Auditing Enforcement Releases (AAER) and the company's own filings before it is treated as ground truth. Tickers can be added freely, and companies that are no longer listed can be added by CIK through `EXTRA_CIKS`.

## Validation

The notebook checks its output in several ways:
- For every company-year with four quarters, the four derived quarters are summed and compared with the reported annual revenue. The mismatch rate is printed.
- Negative revenue, missing revenue and unusual period lengths are marked with `dq_*` columns. These rows are flagged, not dropped.
- `coverage_report.csv` shows the share of quarters with revenue, cash-flow and receivables data for each company.

The pipeline was tested end to end against simulated EDGAR responses (`tests/test_fixture_generator.py`). The simulation includes:
- 10-Q year-to-date reporting and 10-K annual comparatives
- a 52-week fiscal year
- an ASC 606 revenue-tag switch
- a restated quarter
- a spin-off that recasts prior years
- a full-year value mis-tagged as a quarter
- a planted revenue/receivables spike

Results: every derived quarter reconciled to annual revenue, the recast no longer distorted Q4, and the mis-tagged quarter was removed. The restatement and the planted spike still ranked among the top anomalies.

On the real 26-company run, the first version reported a 6.9% quarter-vs-annual mismatch. The top anomalies included impossible values (3M Q4 2022 revenue of $11M; Oracle FY2018 Q4 equal to the full year). These led to the point-in-time fix and the annual sanity check.

To repeat the offline test, run `python tests/test_fixture_generator.py edgar_cache` from the repo root, then run the notebook. The cache makes it read the simulated files. Delete `edgar_cache/` before switching to real data.

## Known limitations

- **Coverage:** Only US-GAAP facts in USD are extracted. Foreign private issuers filing 20-F under IFRS (for example Luckin Coffee) are not covered.
- **Tags not captured:** Custom company-specific tags and segment-level (dimensional) facts are not included in the companyfacts API. When a company reports revenue only under a custom tag, revenue will be missing; `coverage_report.csv` shows this.
- **Banks and insurers:** Their income statements do not map well to revenue/COGS. They are best modelled separately or with sector-specific items.
- **Q4 behaviour:** Derived Q4 figures absorb year-end adjustments, such as audit adjustments and impairments. Q4 therefore shows more legitimate volatility than other quarters, and the anomaly models should treat `fiscal_quarter` as a feature or control for it.
- **Legitimate causes of jumps:** Acquisitions, divestitures, accounting-standard changes and tag switches also cause large deviations. A high Z-score marks a quarter worth reviewing, not evidence of fraud.
- **Recasts count as restatements:** A company that spins off a business, like Honeywell, recasts prior years. These recasts show up as restatements (Honeywell has 25 flagged quarters) even though nothing was misstated. Use `any_amended` or the size of `revenue_restatement_pct` to separate recasts from error corrections.
- **Structural breaks:** Because values are point-in-time, the quarter after a spin-off shows a genuine drop in year-over-year growth, since the prior-year figure still includes the divested business. Examples include J&J after Kenvue (2023), 3M after Solventum (2024) and Honeywell (2025). These are legitimate events, not fraud.
- **Receivables coverage:** Some companies (for example PepsiCo, McDonald's and Target) tag receivables with less common elements. Two additional tags were added; check `coverage_report.csv` after running.
- **Restatement flag scope:** The flag catches restatements that appear as changed comparative figures. Full restatements filed only as 8-K Item 4.02 notices are not captured. Adding 8-K parsing from `/submissions` would be a natural extension.

## Handoff to Deliverable 2 (modelling)

Recommended model inputs are the ratio, growth, earnings-quality and restatement columns. Raw USD amounts depend on company size and should be left out or scaled. The trailing `z_*`/`rz_*` columns serve directly as the Z-score baseline. For Isolation Forest, use company-standardised features, drop rows where `revenue_tag_changed` is true or `dq_*` is set, and compare the resulting anomaly scores between the reference and control groups.

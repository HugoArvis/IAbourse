# CLAUDE.md — IAbourse

## Agent Guidelines (Original Rules — Never Override)

- Always write robust and scalable code
- Keep code simple and avoid unnecessary complexity
- Use clear and consistent naming conventions
- Comment briefly in English when necessary (no over-commenting)
- Follow best practices for the chosen language
- Always test code after writing or modifying it
- Verify that changes do not break existing functionality
- If something is uncertain, explicitly mention it
- Log recurring or important mistakes in CLAUDE.log
- Avoid repeating the same mistakes
- NEVER delete or overwrite important files without explicit permission
- Ask questions if any requirement is unclear
- Do NOT make assumptions or invent missing details
- Do not hallucinate APIs, libraries, or features
- Do not generate untested or speculative code

### Code style

- All Python code (`.py` files and notebook code cells) follows PEP 8
  (https://peps.python.org/pep-0008/): lines <= 79 characters,
  docstrings and comments <= 72, imports grouped stdlib / third-party /
  local, docstring summary on one line, closing quotes of multi-line
  docstrings on their own line, no assigned lambdas.
- Format, then check, before every commit (settings in `setup.cfg` and
  `pyproject.toml`, run from the repo root):
  ```
  python -m black <files>
  python -m isort <files>
  python -m pycodestyle <files>
  ```
  black and isort do not rewrap docstrings and comments: do it by hand.
- Project imports placed after `sys.path.insert(...)` carry
  `# noqa: E402`; library imports go above it.

### How decisions are made on this project

- The project owner decides the strategy. Explain the options and give a
  recommendation, but do not decide for them.
- Work step by step: propose and discuss each step before implementing it, and do
  not start the next phase until asked.

---

## Project Overview

A machine-learning stock selection system on the S&P 500. Every week it ranks the
index members by expected return relative to the other members, holds the top of
the ranking (long-only), and trades through Alpaca (paper trading first).

Goals:
1. Gain solid, hands-on machine-learning experience.
2. Grow a small personal capital (EUR) by reinvesting the gains (compounding).
3. Beat hand-made rules (non-ML factor strategies) by at least a few % per year,
   after costs. Beating the index by a few % per year is the realistic target.

The project was restarted in November 2026 on new data foundations. The old system
(TFT + FinBERT + PPO, per-stock conviction score 0–10) is abandoned: see
"Legacy code" below.

---

## Strategy (decided November 2026)

### Universe and data
- **S&P 500, point-in-time membership**: on each date, only the stocks that were
  in the index on that date, including those that later left it (no survivorship
  bias).
- **Training data starts in 2010.**
- **Free data only** for now. The data layer must stay swappable (one module per
  source in `data_pipeline/sources/`) so that a paid API can replace it later.

### Labels
- For each date, compute every member's forward return over the horizon, minus the
  cross-sectional mean of the members, and split the members into **terciles**:
  0 = worst third, 1 = middle third, 2 = best third.
- Terciles are relative: subtracting a common benchmark (SPY, RSP) does not change
  them. The old ±1.5% vs SPY rule is dropped (unstable class shares).
- **Two horizons: 5 and 21 trading days.** The daily horizon is too noisy.
- **Returns are measured open to open**: from the open of t+1 (when orders are
  executed) to the open of t+1+h, adjusted for dividends and splits.
- Label rules (decided 2026-11-28, `data_pipeline/features/labels.py`):
  members on t with a price on t and t+1; a series ending inside the window
  (acquisition, bankruptcy) earns its return up to the last close, then cash at 0%;
  windows containing a spin-off / special-dividend day get no label; labels on
  every trading day from 2010 (the panel's pre-2010 rows only hold the history of
  later members: biased).

### Timing convention
- Decision on day t uses **all data up to the close of day t** (end-of-day data,
  same sources as the training history). The job runs at night.
- Orders are executed at the **open of t+1**, in the opening auction (Alpaca
  "opg" orders, to be verified). This is 15:30 Paris time.
- The backtest also compares the variant "decide on t, execute at the close of t+1".

### Models
- **LightGBM, CatBoost, XGBoost** (gradient-boosted trees), multiclass on the
  terciles. The score that ranks stocks (e.g. P(best) − P(worst)) is chosen in
  phase 2.
- One model per horizon. The two rankings are combined by a **weighted average of
  ranks**; the weights are chosen on the validation period, never on the test.
- Validation: **walk-forward only**, with purging and an embargo of at least the
  label horizon between training and validation (labels overlap in time).
- Dropped: PPO (reinforcement learning), TFT. FinBERT may come back later only to
  score SEC filing text, compared against Loughran-McDonald word lists.

### Portfolio
- **Long-only, 50 stocks, equal weight** (20 / 50 / 100 compared in backtest).
  A long-short portfolio is computed only as a diagnostic of the signal.
- **Weekly rebalance with a buffer zone**: buy a stock when it enters the top 50,
  sell it when it falls out of about the top 100. This cuts turnover and costs.
  Backtest compares weekly / monthly, with and without buffer.
- Gains are reinvested. A small capital needs fractional shares (Alpaca supports
  them; check per stock).

### How success is measured — benchmark ladder
Every backtest reports, on the same dates and with the same costs:
1. the index: RSP (equal weight) and SPY;
2. single-factor rules: momentum, low volatility, short-term reversal;
3. a multi-factor rule: average of the factor ranks;
4. a linear model (logistic / ridge) on the same features as the trees;
5. each GBDT, then the two horizons combined.

**Success = the ML portfolio beats the multi-factor rule after costs, consistently
across years**, not only on average.
If ML ≈ multi-factor rule on free S&P 500 data, the next step is paid data: it
both fills the remaining gaps of the S&P 500 history and allows a wider universe
(a wider universe needs paid data too: free sources lack prices of delisted
small caps).

### Risk rules (to implement in the new execution code — never bypass)
1. **Stop on relative underperformance**: stop the model if the portfolio lags
   the index by more than a threshold over a rolling window (threshold to define).
   Replaces the old −20% absolute stop.
2. **Max 3–5% of the portfolio per stock.**
3. **Kill switch**: liquidate everything and stop, with explicit confirmation.
4. **At least 6 months of paper trading** before any real money.

Removed from the old rules: confidence filter 0.55 (meaningless with relative
labels), volume filter (all S&P 500 members are liquid), earnings blackout.

### Still open (decide with the owner when the step comes)
- Transaction cost assumptions for the backtest (spread + slippage per side).
- Rebalance weekday (backtest should check that results do not depend on it).
- Threshold and window of the relative stop.

---

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 0. Data sources | PIT membership (Vanguard 500 holdings), validated price panel | Done |
| 1. Exploration | `notebooks/01_exploration_prix_cibles.ipynb`: bias, labels, IC | Done |
| 2. Feature engineering | labels, price/volume features, IC notebook 02, linear + LightGBM baselines with purged CV; then PCA residuals / idiosyncratic vol, peer groups (clustering, SIC); then XBRL fundamentals, macro, short interest, SEC text | Next |
| 3. Models + backtest | 3 GBDTs, horizon combination, full benchmark ladder | |
| 4. Paper trading | new execution code + new dashboard/API, ≥ 6 months | |
| 5. Paid data | only if phase 3 shows ML ≈ multi-factor rule | Conditional |

Feature rules: normalize per date (cross-sectional ranks); fit any scaler, PCA or
clustering on training data only; store features in float32, built year by year
into parquet files.

---

## Data Pipeline (`data_pipeline/`)

```
data_pipeline/
├── sources/        # one module per external source (swappable)
│   ├── sec.py      # EDGAR: filings, XBRL, archives (10 req/s, generic User-Agent)
│   └── tiingo.py   # Tiingo prices + metadata (free quota ~50 requests/hour)
├── membership/     # point-in-time S&P 500 membership
│   ├── vanguard_500.py      # Vanguard 500 Index Fund holdings (N-Q, NPORT-P) = reference
│   ├── wiki_sources.py      # hanshof + index-constitution (Wikipedia-derived, cross-check)
│   ├── entity_match.py      # company name -> CIK
│   ├── tickers.py           # ticker -> CIK
│   ├── successors.py        # same company, new CIK (8-K12B + 1:1 price continuity)
│   └── build_membership.py  # -> cache/sp500_membership.parquet (company_id = grouped CIKs)
├── prices/
│   ├── candidates.py   # candidate tickers per company
│   ├── download.py     # yfinance download of the candidates
│   ├── fill_tiingo.py  # fill gaps from Tiingo (resumable, sleeps on quota)
│   ├── validate.py     # unadjusted close vs Vanguard implied price (3% tolerance)
│   └── panel.py        # -> cache/price_panel.parquet (status validated / extended)
├── features/
│   └── labels.py   # -> cache/labels.parquet (terciles 5d / 21d)
├── tests/          # run each file with python (no pytest needed)
├── audits/         # coverage and consistency checks
├── cache/          # all downloads and outputs (git-ignored, never re-download)
└── requirements.txt
notebooks/          # exploration notebooks (kernel "iabourse")
```

Build order: `membership/vanguard_500.py` → `membership/build_membership.py` →
`prices/candidates.py` → `prices/download.py` → `prices/fill_tiingo.py` →
`prices/validate.py` → `membership/successors.py` → re-run `build_membership.py`,
`candidates.py`, `validate.py` → `prices/panel.py` →
`features/labels.py`.
Run as `python data_pipeline/<folder>/<script>.py` from the repo root.

Price panel facts:
- `ret` = daily total return (close to close). It is NaN on spin-off /
  special-dividend days (`corp_action`), which free data get wrong.
- Price levels (`open`…`close`, `adj_close`) may jump where the series switches
  ticker or source: always chain returns, never levels across a switch.
- `status`: `validated` = checked against Vanguard; `extended` = same series
  outside the checked runs (e.g. after the last Vanguard snapshot).
- Coverage of member-days (validated): ~83% in 2010, ~98% from 2019 on.
  Remaining gaps need paid data: reused tickers (STI, APC, SPLS, EMC, MON…) and
  tickers missing from Tiingo (LLL, HAR, HOT, BEAM, WIN, FRC, SBNY…).

Planned data sources (free): SEC EDGAR XBRL fundamentals (with `filed` dates),
filing text (8-K/10-K/10-Q), Form 4; FRED/ALFRED, CBOE, Fama-French for macro;
FINRA short interest (use the publication date).
Known gaps of free data: delisted-stock prices, historical financial news,
analyst consensus, implied volatility.

---

## Data Leakage Rules (critical — always verify)

- The universe on date t = index members **on date t**, never today's members.
- Features for decision date t use data available **at or before the close of t**
  (16:00 NY). The label starts at the open of t+1, so the close of t is allowed.
- Every non-price input is keyed by the time it became **public**:
  SEC filings by acceptance time (accepted after 16:00 on t → usable from t+1),
  macro series by release date (ALFRED vintages, not revised values),
  short interest by publication date.
- Labels (open t+1 → open t+1+h) never enter features.
- Validation and test periods are always after training periods; purge and
  embargo ≥ the label horizon.
- Anything fitted (scalers, PCA, clusters, ensemble weights, hyperparameters) is
  fitted on training / validation data only, never on the test period.

---

## Hardware Constraints

- Development machine: **5 GB RAM**. An out-of-memory error already killed VS Code.
  Run heavy scripts under a memory cap:
  `systemd-run --user --scope -q -p MemoryMax=1500M -p MemorySwapMax=0 python ...`
- Target hardware: a cheap NUC with 8–16 GB. Design every job for that: float32,
  year-by-year processing written to parquet, day subsampling for tuning.
- Heavy jobs can run on free Kaggle notebooks. Never put API keys or `.env`
  content in a Kaggle notebook.

---

## Environment Variables (`backend/.env` — never commit, never print)

```
TIINGO_API_KEY=...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

SEC requests use a generic User-Agent (`data_pipeline/sources/sec.py`). Never put
personal details (name, email) in requests to external services.

---

## Legacy Code (obsolete — do not extend)

`backend/` (FastAPI, scheduler at 15:30 NY, TFT/PPO training, per-stock score
0–10, old risk manager) and `frontend/` (React dashboard) belong to the old system.
They do not match the strategy above. The execution code, the API and the
dashboard will be **redesigned** in phase 4; the new dashboard must show what
matters for testing (positions vs ranking, turnover, costs, performance vs the
benchmark ladder, data freshness). Until then, only `backend/.env` is used by the
new pipeline. Do not delete legacy files without permission.

---

## Git Workflow (mandatory)

- `main`: stable releases only. Never commit or push directly to `main`.
- `dev`: integration branch. Never commit directly to `dev` either.
- Every feature or fix gets its own branch created from `origin/dev`:
  `feature/<short-name>`, `fix/<short-name>` or `docs/<short-name>`.
- Feature branches are merged into `dev` through a Pull Request.
  **Check that the PR base is `dev`**: the GitHub default branch is `main`, and
  the ruleset cannot block a PR opened against `main` by mistake.
- When `dev` reaches a good, tested state, open a PR `dev` → `main` for the release.
- Never force-push to `main` or `dev`. Never rewrite their history.

---

## Key Decisions Log

| Decision | Reason |
|---|---|
| Restart on new data foundations (Nov 2026) | Old pipeline had survivorship bias and constant neutral news history |
| Vanguard 500 holdings as membership reference | Wikipedia-derived lists miss 30–37% of members in 2004 and contain errors |
| Prices validated against Vanguard implied price | Catches reused tickers and wrong series |
| Training from 2010 | ~16 years is enough for GBDTs; free coverage weaker before |
| Cross-sectional tercile labels | Stable class shares; independent of the benchmark |
| Horizons 5d + 21d combined | Daily is too noisy; two horizons strengthen the signal |
| Decide after close t, execute at open t+1 | Uses the most recent day; same EOD data in training and live; easy to automate from Paris |
| GBDTs (LightGBM, CatBoost, XGBoost) | Strong on tabular data, fit the hardware |
| PPO and TFT dropped | Too complex for the signal available; ranking + simple portfolio rules instead |
| Long-only top 50, weekly with buffer | Simple, low turnover, works with a small account |
| Success = beat the multi-factor rule after costs | Beating the index alone does not prove ML adds value |
| News from SEC filings, not Finnhub | Free history with exact publication times |
| Free data first, paid data only if needed | Bottleneck is signal/features, not prices; paid data also needed for a wider universe |

---

## CLAUDE.log Format

Append errors here. Format: `[DATE] [FILE] [TYPE] — description + fix applied`

Example:
```
[2026-05-05] [trading_env.py] [BUG] — reward NaN when rolling_std=0 on first 20 days.
  Fix: added epsilon=1e-8 denominator in sharpe_incremental calculation.
```

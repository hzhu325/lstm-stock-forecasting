# LSTM Stock Price Prediction

A PyTorch LSTM model that predicts next-day closing prices from a 60-day
sliding window of historical closing prices. The project is a small,
self-contained exercise in time-series forecasting with deep learning, with
particular attention to common methodological pitfalls (look-ahead leakage,
recursive vs. rolling-window forecasting, and the limits of a single-feature
model).

This project was developed as part of my data science internship at BONC
(东方国信), April – June 2026. The full methodology, results and reflections
are documented in `project_report.md`.

---

## Highlights

- **No look-ahead leakage**: `MinMaxScaler` is fitted on the training portion
  only, then applied to the whole series. The 80/20 train/test split is
  strictly chronological — no shuffling.
- **Rolling-window forecasting, not recursive**: for the recent-30-day
  forecast, every prediction uses real historical inputs. An earlier draft
  fed predictions back into the model recursively; that was corrected after
  rechecking what the task actually required.
- **Model comparison**: a baseline (`hidden=50`) is compared against a
  regularised variant (`hidden=64`, `dropout=0.2`) on MSE, RMSE and MAE.
- **Multi-source data fetching**: `akshare` is the primary data source
  (A-share daily bars). `yfinance` is used as a fallback when the akshare
  API is unavailable or rate-limited.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Running `main.py` will:

1. Download daily bars for the configured stock code (Ping An Bank,
   `000001`) over `2020-01-01` to `2025-01-01`.
2. Fit `MinMaxScaler` on the training portion and transform the whole series.
3. Train a baseline two-layer LSTM (`hidden=50`).
4. Train a regularised variant (`hidden=64`, `dropout=0.2`).
5. Run a rolling-window forecast over the most recent 30 trading days.
6. Save evaluation plots into `figures/`.

## Results

| Model | Test MSE | Test RMSE | Test MAE |
| --- | --- | --- | --- |
| `lstm_base` (hidden=50) | 0.3023 | 0.5498 | 0.5039 |
| `lstm_hidden64_dropout` (hidden=64, dropout=0.2) | 0.3015 | 0.5491 | 0.4049 |

The regularised model improves MAE noticeably while leaving MSE/RMSE largely
unchanged. The recent-30-day rolling forecast tracks the overall trend but
predicts noticeably above the actuals — a known limitation of using
closing-price-only inputs.

See `project_report.md` for the full discussion, including the figures and
the methodology corrections that happened during the project.

## Limitations

This is a coursework-scale experiment, not a trading system.

- Single feature (closing price). OHLCV, fundamentals, macro and news
  features are out of scope.
- One stock symbol. Generalisation across symbols is not evaluated.
- No dedicated validation set; hyperparameters were chosen by report-time
  reflection, not by systematic search.
- The model produces a smoothed trajectory that lags real movement —
  expected behaviour for an MSE-trained, single-feature LSTM, but worth
  stating plainly.

Possible extensions are listed at the end of `project_report.md`: multi-
feature inputs, GRU / Transformer architectures, and a proper validation
split for hyperparameter tuning.

## Acknowledgements and Development Notes

This project was written with the help of AI-assisted coding. The dataset
choice, the 60-day window, the train/test discipline (no look-ahead), the
model comparison and the switch from recursive to rolling-window forecasting
were design decisions I made and worked through; the code was produced by
iterative prompting, manual review and integration testing on my side.

`project_report.md` is the most honest description of what worked, what did
not, and what I learned during the project — including the methodological
correction on the rolling-window forecast. If you are reading this as part
of an application review, that report is the document I would point you to
first.

---

Author: Hongtao Zhu · hzhu325@wisc.edu · github.com/hzhu325

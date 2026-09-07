# F1 constructors forecast

The original coursework and preseason analysis remain in `finalproject.ipynb`.
`refresh_forecast.py` adds a reproducible in-season model for the portfolio desktop.
The latest export is through the 2026 Italian Grand Prix, checked September 7.

## Refresh

Requires Python, numpy, pandas and scikit-learn. Tested package versions are pinned
in `requirements-forecast.txt` (Python 3.14).

```sh
python refresh_forecast.py --season 2026 --as-of 2026-09-07 --refresh
python -m unittest test_refresh_forecast.py
```

Omit `--refresh` to reproduce the checked-in input snapshot without network calls.
The output is `data/f1-forecast.json`. The portfolio imports this export. Refreshing
this repository does not automatically deploy or edit the portfolio.

## In-season v2 methodology

- Historical race-level features come from the existing 2010–2025 clean results CSV.
- For each historical season, use only the first floor(season race count × current
  season completion fraction) Grands Prix. Current inputs include all 13 completed
  2026 Grands Prix out of the current 23-race calendar.
- Features: race-only points share and points rank, mean grid/finish position,
  retirement rate, wins/podiums/top-ten finishes per driver entry. Sprint points
  are not included in feature aggregates, consistently across historical and
  current seasons. Displayed current championship points include Sprints and
  official deductions.
- Final targets are official championship ranks from Jolpica, including Sprint
  points and adjustments, rather than ranks inferred from race-only point totals.
- Exclude 2018: the legacy CSV merges the distinct Force India championship entries.
  Other constructor names are mapped explicitly, with missing mappings rejected.
- Lapped classified finishes are not counted as retirements. Pit-lane grid zeroes
  are treated as missing grid positions. Numeric missing values are median-imputed
  inside each training fold.
- Standardized Ridge regression with fixed alpha 10. Expanding-window validation
  begins after five training seasons. Every evaluation season is predicted from
  earlier seasons only; no random team-row split or future-season preprocessing.
- Fit the final model on all eligible completed historical seasons. Sample 10,000
  independent draws from the held-out rank residuals, add them to predicted ranks,
  and rank all current constructors within each draw. Seed 42. The export contains
  title, top-three, top-five frequencies and mean simulated rank.

This reconstructs the missing July in-season pipeline; it does not claim to be
the identical historical model. Validation errors are rank-prediction errors,
not probability calibration scores. Estimated probabilities are not betting odds.
Residual exchangeability, independence across teams, varying grid sizes, and new
regulations/entrants limit reliability. No current-season final result is a target.
The naive baseline predicts final rank using the race-points rank at the cutoff.

## Sources and saved inputs

[Jolpica API](https://api.jolpi.ca/ergast/f1/) supplies paginated current results,
calendar, Sprints, driver standings and official historical/current constructor
standings. `data/inputs-2026.json` preserves the exact responses used for this export.
The scraper honors page limits, uses request timeouts, and stops on network errors;
it does not continuously poll or retry rate-limited requests.

Current points and leader were cross-checked against the official
[constructor standings](https://www.formula1.com/en/results/2026/team),
[driver standings](https://www.formula1.com/en/results/2026/drivers), and
[calendar](https://www.formula1.com/en/racing/2026).

Only data and reproducible source are published; no credentials are required.

# FusionFinance Filing Alpha Transforms

This package contains a deliberately narrow port of pure, auditable transforms
from the supplied `filing_alpha_sharpe6` archive (`filing-alpha-cleanroom`
0.6.0) under the MIT License.

Included:

- dataframe validation and normalization;
- XBRL concept aliases, ratios, and longitudinal filing deltas;
- deterministic filing-text statistics and change features;
- point-in-time session mapping, as-of joins, freshness decay, and robust
  cross-sectional standardization;
- dependency-light portfolio metrics and the transparent ridge fusion baseline.

Excluded on purpose:

- the raw SEC acceptance parser, because it localized EDGAR's timezone-naive
  timestamp as UTC and could shift availability by four to five hours;
- download/network code, generic model/training code, calibration grids,
  portfolio code, and legacy strategies already superseded by FusionFinance;
- all claimed trading logic from the frozen Sharpe6 candidate.

The source archive itself rejects that candidate: locked 2016 Sharpe was
`-0.430`, 2024 transfer Sharpe was approximately `0.044`, estimated PBO was
`68.57%`, and deflated-Sharpe probability was `5.71%`. These transforms are
software building blocks, not evidence of profitable alpha.

Acceptance timestamps passed to `map_filings_to_available_sessions` must be
timezone-aware. Its same-session mapping means "known by the after-close signal"
and never authorizes a same-close execution fill.

See `NOTICE` for attribution and license terms.

# Contributing to FusionFinance

FusionFinance welcomes reproducible improvements to its research, execution
kernel, verification layer, and demo. Contributions must preserve the
project's central rule: language-model output cannot reach capital without
evidence, timing, accounting, and risk checks.

## Choose the right path

- Submission-critical code lives in `demo/`, `alpha/verifier/`,
  `alpha/filing_alpha/`, `configs/`, `evidence/`, and the focused tests.
- Keep exploratory research outside the public release unless it closes a
  documented architecture gap and ships with evidence and tests.
- Start with the [repository guide](docs/README.md) before moving files or
  changing artifact formats.

## Development workflow

1. Create a short-lived branch from `main`; keep `main` deployable.
2. Add a failing test that captures the intended behavior.
3. Implement the smallest change that makes it pass, then refactor.
4. Rebuild the artifacts and run the focused release suite.
5. Document any change to timing, costs, data availability, metrics, or
   claim status in the methodology and limitations.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'

make artifacts
make verify
make test
```

Use conventional commit subjects such as `feat:`, `fix:`, `test:`, and
`docs:`. Never commit API keys, cookies, account identifiers, private market
data, or generated caches.

## Research integrity

Every performance-facing contribution must state:

- the exact data and availability timestamps;
- the decision cutoff and executable price;
- costs, slippage, leverage, and position limits;
- whether the run was preregistered, retrospective, or prospective;
- all failed variants relevant to the reported result;
- limitations that could change the conclusion.

Do not optimize a window after observing its result and relabel it as
out-of-sample. Do not turn a replay into a causal claim. A failing evidence
or lineage check must fail closed.

## License

By submitting a contribution, you agree that your original contribution is
licensed under GPL-3.0-only. Third-party material must be compatible with the
repository and retain its own license and attribution in `NOTICE` or the
relevant subdirectory.

# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/aben00/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/aben00/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day, 2026-11-19 (UTC), from the first hour's viewer
count. A make.fun `program/v1` entry (Python 3.13, standard library only,
no network, deterministic). Bounty:
https://make.fun/bounty/gta6-twitch-ccu-forecast-01

## Approach

The case anchors on its hour-1 print `p1` and forecasts hours 2..24 as five
quantiles each. The case ships `comparables` — the hourly Twitch curves of
past big game launches — and for each one we take its release-day shape as
log ratios `ln(viewers_h / viewers_1)`. Two views of that shape are blended
(geometric mean of quantiles):

1. **Kernel-weighted empirical quantiles.** Each comparable is weighted by
   a Gaussian kernel on `|ln p1_i - ln p1|` (bandwidth 1.0 in log space),
   so launches of a similar size dominate. This matters: across past
   launches the log ratio is strongly negatively correlated with `log p1`
   (about −0.9 at hour 18) — a launch whose category was nearly empty at
   00:00 UTC can explode 100× by the US unlock, while a launch already at
   100k+ viewers in hour 1 typically only doubles.
2. **Per-hour regression.** OLS of the log ratio on `log p1` across
   comparables, with the spread from the empirical quantiles of the
   regression residuals. This extrapolates the size dependence smoothly
   where the kernel has few neighbours.

If an hour has too few usable comparables, it falls back to the house
persistence baseline (every hour stays at `p1`, spread `0.15·√(h−1)` in
log space), so the output is always a valid 23-row forecast.

Local validation is leave-one-out over the 38 practice comparables (the
kit ships no practice answers and its `grade-local` refuses program/v1
kits, so this is the honest local check): pinball loss 0.254 on big
launches (hour-1 ≥ 20k viewers) vs 0.321 for the persistence baseline,
typical miss ~46% vs 63%, 90% coverage ~0.86. Over all comparables the
loss is 0.39 vs 0.93 baseline.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` with
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` — exactly
23 rows, hours 2..24.

## License

MIT — see `LICENSE`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=bh1grvtt&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

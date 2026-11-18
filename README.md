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
log ratios `ln(viewers_h / viewers_1)`. For every hour, separately, the
forecast is built from the comparables that have that hour, in four steps:

1. **Size-weighted local-linear fit.** Each comparable is weighted by a
   Gaussian kernel on `|ln p1_i - ln p1|` (bandwidth 1.25 in log space),
   so launches of a similar size dominate. This matters: across past
   launches the log ratio is strongly negatively correlated with `log p1`
   (about −0.9 at hour 18) — a launch whose category was nearly empty at
   00:00 UTC can explode 100× by the US unlock, while a launch already at
   100k+ viewers in hour 1 typically only doubles. A weighted
   least-squares line of the log ratio on `log p1` is fit at the target's
   size.
2. **Weak OU-style pull.** The local fit's prediction is pulled toward the
   kernel-weighted mean by a factor 0.7 — the size signal is real but
   noisy — and the median is blended 25/75 with the kernel-weighted
   empirical median, which is robust when a few launches sit far from the
   line.
3. **Spread shrinkage.** The spread comes from the empirical quantiles of
   the local fit's residuals, each shrunk toward the normal shape at that
   hour's residual spread, which is itself shrunk toward the pooled
   spread across all 23 hours (prior weight 20). With ~37 curves, one
   hour's 5%/95% quantiles are noisy; the shrinkage borrows strength
   across hours without forcing every hour to share one spread.
4. **Interval calibration.** The half-widths around the median are scaled
   by 1.1, bringing the 90% coverage to ~0.90 — where the pinball loss is
   lowest.

If an hour has too few usable comparables, it falls back to the house
persistence baseline (every hour stays at `p1`, spread `0.15·√(h−1)` in
log space), so the output is always a valid 23-row forecast.

Local validation is leave-one-out over the 37 usable practice
comparables (the kit ships no practice answers and its `grade-local`
refuses program/v1 kits, so this is the honest local check): pinball loss
0.377 pooled / 0.253 on big launches (hour-1 ≥ 20k viewers) / 0.212 on
mega-launches (hour-1 ≥ 100k), vs 0.930 / 0.321 for the persistence
baseline on the same cases; typical miss ~76%; 90% coverage 0.897. The
previous shipped model scored 0.388 / 0.266 on the same check.

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

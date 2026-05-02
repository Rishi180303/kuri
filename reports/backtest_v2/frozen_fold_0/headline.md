# Phase 4 — Primary Backtest Headline

```
==========================================================================================
BACKTEST HEADLINE — frozen_fold_0  (2022-07-04 -> 2026-04-01)
==========================================================================================
series                     CAGR   Sharpe    MaxDD    Alpha   Beta   p(α)
------------------------------------------------------------------------
frozen_fold_0            19.82%     0.86  -17.38%    8.10%   1.06  0.053
nifty50                  10.30%     0.38  -15.77%     nan%    nan    nan
ew_nifty49               19.16%     0.96  -17.79%    8.03%   0.98  0.000
------------------------------------------------------------------------
strategy total cost paid: 111,571 INR (11.157% of initial capital)
strategy n_rebalances:    47
n problematic trades:     764 (see flag_problematic in trade_log)
==========================================================================================
```

## Raw metrics JSON

```json
{
  "config": {
    "name": "frozen_fold_0",
    "backtest_start": "2022-07-04",
    "backtest_end": "2026-04-01",
    "initial_capital": 1000000.0,
    "n_positions": 10,
    "rebalance_freq_days": 20
  },
  "strategy_metrics": {
    "total_return": 0.9404457946413018,
    "cagr": 0.19817042397924078,
    "annualized_vol": 0.1561784028656152,
    "sharpe": 0.8631758288849273,
    "sortino": 0.8345830361926299,
    "max_drawdown": -0.17384544692634626,
    "max_drawdown_duration": 231.0,
    "calmar": 1.139923003351364,
    "alpha_annualized": 0.08099355677584205,
    "beta": 1.0578975896458718,
    "alpha_pvalue": 0.052529540973733635,
    "information_ratio": 1.0886277332833754
  },
  "benchmark_metrics": {
    "nifty50": {
      "total_return": 0.4322008007450695,
      "cagr": 0.10304344324295012,
      "annualized_vol": 0.12691915383031133,
      "sharpe": 0.37723985013248545,
      "sortino": 0.360528907311606,
      "max_drawdown": -0.1576667830374075,
      "max_drawdown_duration": 314.0,
      "calmar": 0.6535520117671353,
      "alpha_annualized": NaN,
      "beta": NaN,
      "alpha_pvalue": NaN,
      "information_ratio": NaN
    },
    "ew_nifty49": {
      "total_return": 0.9014521136326659,
      "cagr": 0.19155528695579815,
      "annualized_vol": 0.1302419969290685,
      "sharpe": 0.96387203603491,
      "sortino": 0.9008477777497731,
      "max_drawdown": -0.17786456627293065,
      "max_drawdown_duration": 280.0,
      "calmar": 1.0769727268884983,
      "alpha_annualized": 0.08026712004109698,
      "beta": 0.9772942752628065,
      "alpha_pvalue": 0.00011251957369196042,
      "information_ratio": 1.9648542366718098
    }
  },
  "n_rebalances": 47,
  "total_cost_inr": 111571.0466218974
}
```

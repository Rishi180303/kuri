# Phase 4 — Primary Backtest Headline

```
==========================================================================================
BACKTEST HEADLINE — primary  (2022-07-04 -> 2026-04-01)
==========================================================================================
series                     CAGR   Sharpe    MaxDD    Alpha   Beta   p(α)
------------------------------------------------------------------------
primary                  27.62%     1.22  -17.49%   14.66%   1.05  0.003
nifty50                  10.30%     0.38  -15.77%     nan%    nan    nan
ew_nifty49               19.16%     0.96  -17.79%    8.03%   0.98  0.000
------------------------------------------------------------------------
strategy total cost paid: 120,496 INR (12.050% of initial capital)
strategy n_rebalances:    47
n problematic trades:     0 (trades with size > 1% of 20d ADV)
==========================================================================================
```

## Raw metrics JSON

```json
{
  "config": {
    "name": "primary",
    "backtest_start": "2022-07-04",
    "backtest_end": "2026-04-01",
    "initial_capital": 1000000.0,
    "n_positions": 10,
    "rebalance_freq_days": 20
  },
  "strategy_metrics": {
    "total_return": 1.4458220369886976,
    "cagr": 0.2762449922530148,
    "annualized_vol": 0.16254931027872807,
    "sharpe": 1.2242707660450989,
    "sortino": 1.1646191585589518,
    "max_drawdown": -0.1748901594889311,
    "max_drawdown_duration": 183.0,
    "calmar": 1.579534223424952,
    "alpha_annualized": 0.146611488325629,
    "beta": 1.045147931495285,
    "alpha_pvalue": 0.0029345086036984824,
    "information_ratio": 1.6093051520245971
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
  "total_cost_inr": 120495.81937973044
}
```

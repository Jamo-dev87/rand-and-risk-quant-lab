# Strategy Framework

This project is based on converting a discretionary US index trading strategy into a rules-based research system.

## Core Idea

The strategy looks for moments where price takes liquidity from one side of the market, confirms direction through a break of structure, and then offers a retracement entry.

## Planned Rules

### Long Setup

1. Higher-timeframe bias is bullish.
2. Price sweeps below the London session low.
3. Price breaks above the most recent structure high.
4. Entry is modelled at the 0.5 retracement of the break-of-structure move.
5. Stop-loss is placed below the swept low.
6. Take-profit is modelled using fixed risk/reward.

### Short Setup

1. Higher-timeframe bias is bearish.
2. Price sweeps above the London session high.
3. Price breaks below the most recent structure low.
4. Entry is modelled at the 0.5 retracement of the break-of-structure move.
5. Stop-loss is placed above the swept high.
6. Take-profit is modelled using fixed risk/reward.

## Current Version

The current dashboard does not yet backtest the full strategy. It uses moving averages as a first proxy for market bias while the more detailed intraday logic is developed.
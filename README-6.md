# Multi-Armed Bandit Portfolio Allocation

Reinforcement learning for dynamic asset allocation. Each week, an agent chooses **one asset to hold** (an "arm"), observes its return as the reward, and updates its beliefs. The core problem is the **exploration vs exploitation** trade-off in a market where the best asset changes with the regime.

> Returns are **simulated** from a regime-switching model, so results are reproducible and the true best asset is known at every date.

---

## Key results

10 years of weekly returns (520 weeks), 5 assets, 5 market regimes.

| Strategy | Total return | Ann. return | Sharpe | Max drawdown | Weeks in best asset |
|---|---:|---:|---:|---:|---:|
| Oracle (knows best asset) | 1,136% | 28.6% | 2.35 | −9.0% | 100% |
| **UCB1** | **281%** | **14.3%** | **1.44** | −15.6% | **37.1%** |
| ε-greedy (ε = 0.1) | 91% | 6.7% | 1.21 | −7.7% | 4.8% |
| Equal weight | 57% | 4.6% | 1.20 | −8.1% | – |
| Discounted Thompson (γ = 0.97) | 99% | 7.1% | 0.80 | −17.6% | 26.7% |
| Thompson sampling | 69% | 5.4% | 0.67 | −14.5% | 22.3% |
| Random | 11% | 1.1% | 0.17 | −25.2% | 17.5% |

**Takeaways**

- **UCB1 performs best**: its uncertainty bonus keeps re-testing assets it has not sampled recently, which helps after regime changes.
- **ε-greedy gets stuck** on an early winner and holds the best asset in only 5% of weeks.
- **Forgetting helps**: discounting old rewards lifts Thompson sampling from 69% to 99% total return and lowers regret.
- **Return is not risk-adjusted performance**: single-asset bets carry larger drawdowns than the diversified equal-weight portfolio.

---

## Market simulation

| Asset | Bull | Bear | Rotation |
|---|---|---|---|
| Equities | Best | Worst | Flat |
| Bonds | Low | Positive | Flat |
| Gold | Low | Best | Low |
| Commodities | Positive | Negative | Best |
| Cash | Risk-free | Risk-free | Risk-free |

Regime schedule: Bull (150 weeks) → Bear (90) → Rotation (110) → Bull (100) → Bear (70).

---

## Agents

| Agent | Selection rule |
|---|---|
| **ε-greedy** | Best estimated mean, random asset with probability ε |
| **UCB1** | $\arg\max_a \; \hat{\mu}_a + c\sqrt{2\ln t / n_a}$, with $c$ scaled to weekly returns |
| **Thompson sampling** | Gaussian conjugate posterior per asset; sample a mean for each, pick the highest |
| **Discounted Thompson** | Same, with past observations discounted by γ each week to adapt to regime changes |

**Regret** at time $t$ is the cumulative gap between the oracle's expected return and the expected return of the asset chosen:

$$
\text{Regret}_T = \sum_{t=1}^{T}\left(\max_a \mu_{a,t} - \mu_{a_t,t}\right)
$$

---

## Quick start

```bash
git clone https://github.com/NourSALHI28/REINFORCEMENT-LEARNING.git
cd REINFORCEMENT-LEARNING
pip install numpy pandas matplotlib jupyter
jupyter notebook mab_portfolio_allocation.ipynb
```

The notebook runs in under a minute.

## Contents of the notebook

1. Regime-switching market simulation
2. Implementation of the four bandit agents
3. Backtest against equal-weight, random and oracle benchmarks
4. Performance table, growth of $1 and cumulative regret
5. Asset choices over time vs the truly best asset
6. Takeaways and limitations

---

## Limitations

- Simulated returns and a single random seed.
- No transaction costs; frequent switching would be penalised in practice.
- One asset held per period rather than a weighted portfolio.
- Results are sensitive to hyperparameters (ε, UCB bonus, discount γ).

## Next steps

Real market data, transaction costs, multiple seeds with confidence intervals, contextual bandits using signals (momentum, volatility), and weight allocation across several assets.

## Author

**Nour Salhi** · [LinkedIn](https://www.linkedin.com/in/nour-salhi-finance) · [GitHub](https://github.com/NourSALHI28)

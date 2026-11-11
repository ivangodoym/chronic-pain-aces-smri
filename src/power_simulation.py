"""
src/power_simulation.py
Monte Carlo power analysis for the indirect effect (a*b) in a simple mediation model.

The pipeline in src/modeling.py tests each indirect effect with a Monte Carlo
confidence interval (Preacher & Selig, 2012). This script asks a companion question:
given standardized paths a and b and a sample size n, how often would that test
detect the indirect effect? Power is estimated by simulating data from the model,
fitting the same two OLS regressions as fit_ols_paths(), and checking whether the
95% Monte Carlo interval for a*b excludes zero.

Usage
-----
    python -m src.power_simulation                       # default grid (see below)
    python -m src.power_simulation --a 0.12 --b 0.11 --n 406
    python -m src.power_simulation --a 0.2 --b 0.2 --n 406 --reps 5000

Paths a and b are standardized (all variables have unit variance), so they can be read
directly from the pooled estimates in outputs/tables/mediation_*_results.csv.
"""
import argparse
import numpy as np
import statsmodels.api as sm

from src.config import RANDOM_STATE

# Default grid: the observed path sizes in this dataset at the two effective sample
# sizes, plus the sizes needed to reach conventional power.
DEFAULT_GRID = [
    (0.12, 0.11, 406),
    (0.12, 0.11, 141),
    (0.20, 0.20, 406),
    (0.25, 0.25, 141),
]


def simulate_power(a: float, b: float, n: int, reps: int = 2000, n_mc: int = 5000,
                   alpha: float = 0.05, seed: int = RANDOM_STATE) -> float:
    """Return the proportion of simulated datasets in which the Monte Carlo CI for a*b excludes 0.

    Data are generated as X ~ N(0, 1); M = a*X + e_M; Y = b*M + e_Y, with residual
    variances chosen so that M and Y also have unit variance (no direct effect c').
    """
    rng = np.random.default_rng(seed)
    sd_m = np.sqrt(1.0 - a ** 2)
    sd_y = np.sqrt(1.0 - b ** 2)
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    hits = 0

    for _ in range(reps):
        x = rng.normal(size=n)
        m = a * x + rng.normal(scale=sd_m, size=n)
        y = b * m + rng.normal(scale=sd_y, size=n)

        fit_a = sm.OLS(m, sm.add_constant(x)).fit()                 # path a
        fit_b = sm.OLS(y, sm.add_constant(np.column_stack([x, m]))).fit()  # paths c' and b

        draws_a = rng.normal(fit_a.params[1], fit_a.bse[1], size=n_mc)
        draws_b = rng.normal(fit_b.params[2], fit_b.bse[2], size=n_mc)
        ci_low, ci_high = np.percentile(draws_a * draws_b, [lo_q, hi_q])
        hits += (ci_low > 0) or (ci_high < 0)

    return hits / reps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=float, help="standardized path a (X -> M)")
    parser.add_argument("--b", type=float, help="standardized path b (M -> Y | X)")
    parser.add_argument("--n", type=int, help="sample size")
    parser.add_argument("--reps", type=int, default=2000, help="simulated datasets per scenario (default 2000)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    grid = [(args.a, args.b, args.n)] if args.a is not None else DEFAULT_GRID
    if args.a is not None and (args.b is None or args.n is None):
        parser.error("--a, --b and --n must be given together")

    print(f"{'a':>6} {'b':>6} {'n':>6} {'power':>7}   (reps = {args.reps}, seed = {args.seed})")
    for a, b, n in grid:
        power = simulate_power(a, b, n, reps=args.reps, seed=args.seed)
        print(f"{a:>6.2f} {b:>6.2f} {n:>6d} {power:>7.2f}")


if __name__ == "__main__":
    main()

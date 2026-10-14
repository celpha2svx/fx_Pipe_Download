import numpy as np
from pathlib import Path
import logging

from lox.engine.hurst import compute_hurst

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fgn_hosking(n: int, H: float, seed: int) -> np.ndarray:
    """
    Generate exact fractional Gaussian noise (fGn) via Hosking's method.

    fGn is the increment process of fractional Brownian motion.
    It has a TRUE, scale-invariant Hurst parameter H:
        H = 0.50 -> iid (random walk increments)
        H > 0.50 -> persistent (positive long-range correlation)
        H < 0.50 -> anti-persistent (negative correlation)

    Unlike AR(1), fGn has NO crossover: its correlation is a power law,
    so DFA measures the same H at every scale. This is the correct
    ground truth for calibrating a Hurst estimator.
    """
    rng = np.random.default_rng(seed)

    # fGn autocovariance: gamma(k) = 0.5*(|k-1|^(2H) + |k+1|^(2H) - 2|k|^(2H))
    k = np.arange(n, dtype=float)
    gamma = 0.5 * (np.abs(k - 1) ** (2 * H)
                   + np.abs(k + 1) ** (2 * H)
                   - 2 * np.abs(k) ** (2 * H))
    gamma0 = gamma[0]  # = 1.0

    z = rng.standard_normal(n)
    x = np.zeros(n)
    x[0] = np.sqrt(gamma0) * z[0]

    # phi[j] holds the order-current coefficient for lag j (index 0 unused)
    phi = np.zeros(n + 1)

    for step in range(1, n):
        if step == 1:
            phi_kk = gamma[1] / gamma[0]
        else:
            # Previous row coefficients live in phi[1:step]
            num = gamma[step] - np.dot(phi[1:step], gamma[step - 1:0:-1])
            den = gamma[0] - np.dot(phi[1:step], gamma[1:step])
            phi_kk = num / den

        # Recursively update the coefficient row in place
        if step > 1:
            prev = phi[1:step].copy()
            phi[1:step] = prev - phi_kk * prev[::-1]
        phi[step] = phi_kk

        # Innovation variance (clamped for floating-point safety)
        innov_var = max(gamma0 * (1.0 - np.dot(phi[1:step + 1], phi[1:step + 1])), 0.0)

        # X[step] = sum_j phi_{step,j} * X[step-j] + innovation
        x[step] = np.dot(phi[1:step + 1], x[step - 1::-1]) + np.sqrt(innov_var) * z[step]

    return x


def validate_hurst_engine():
    """
    Part B (corrected): Validate the DFA engine against fGn with known H.

    Protocol:
      - 10,000 points per realization
      - 5 independent seeds per case (averaged to remove sampling noise)
      - DFA scales 8..500 (log-spaced)
      - Acceptance: |mean_measured_H - true_H| <= 0.05
    """
    n_points = 10000
    seeds = [42, 123, 777, 2024, 99]

    # (label, true Hurst parameter)
    cases = [
        ("Random Walk / fGn H=0.50", 0.50),
        ("Persistent / fGn H=0.75", 0.75),
        ("Anti-persistent / fGn H=0.25", 0.25),
    ]

    results = []
    all_passed = True

    logging.info("=" * 60)
    logging.info("PART B: HURST ENGINE VALIDATION (DFA vs fGn)")
    logging.info("=" * 60)

    for name, true_H in cases:
        logging.info(f"\nTesting: {name}")
        measured = []

        for seed in seeds:
            returns = fgn_hosking(n_points, true_H, seed)
            h = compute_hurst(returns, min_window=8, max_window=500, num_windows=20)
            measured.append(h)
            logging.info(f"  seed={seed:<5} measured H={h:.4f}")

        mean_H = float(np.mean(measured))
        std_H = float(np.std(measured))
        error = abs(mean_H - true_H)
        passed = error <= 0.05

        logging.info(f"  Mean H:    {mean_H:.4f}  (std={std_H:.4f})")
        logging.info(f"  True H:    {true_H:.4f}")
        logging.info(f"  Error:     {error:.4f}")
        logging.info(f"  Status:    {'PASS' if passed else 'FAIL'}")

        results.append({
            "name": name,
            "true_H": true_H,
            "measured": measured,
            "mean_H": mean_H,
            "std_H": std_H,
            "error": error,
            "passed": passed,
        })

        if not passed:
            all_passed = False

    # Write auditable report
    Path("results").mkdir(exist_ok=True)
    report_file = Path("results") / "hurst_validate_report.txt"

    with open(report_file, "w") as f:
        f.write("HURST ENGINE VALIDATION REPORT (DFA vs fGn)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Method: Detrended Fluctuation Analysis (DFA)\n")
        f.write("Ground truth: fractional Gaussian noise (Hosking, exact)\n")
        f.write(f"Points per realization: {n_points}\n")
        f.write(f"Seeds per case: {len(seeds)} (averaged)\n")
        f.write("DFA scale range: 8..500 (log-spaced)\n")
        f.write("Acceptance: |mean_measured_H - true_H| <= 0.05\n\n")

        for r in results:
            f.write(f"Test: {r['name']}\n")
            f.write(f"  True H:       {r['true_H']:.4f}\n")
            f.write(f"  Per-seed H:   {', '.join(f'{h:.4f}' for h in r['measured'])}\n")
            f.write(f"  Mean H:       {r['mean_H']:.4f}\n")
            f.write(f"  Std H:        {r['std_H']:.4f}\n")
            f.write(f"  Error:        {r['error']:.4f}\n")
            f.write(f"  Status:       {'PASS' if r['passed'] else 'FAIL'}\n\n")

        f.write("=" * 60 + "\n")
        if all_passed:
            f.write("OVERALL RESULT: ALL TESTS PASSED\n")
            f.write("Engine is validated. Proceed to forex deployment (Part C).\n")
        else:
            f.write("OVERALL RESULT: SOME TESTS FAILED\n")
            f.write("Do NOT proceed. Inspect per-seed spread to separate\n")
            f.write("sampling noise from systematic estimator bias.\n")

    logging.info(f"\nReport saved to: {report_file}")
    logging.info("\n" + ("VALIDATION PASSED. Ready for forex." if all_passed
                         else "VALIDATION FAILED. Do not proceed."))

    return all_passed, results


if __name__ == "__main__":
    validate_hurst_engine()
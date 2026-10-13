"""Generate figures for the ER-CCB paper from experiment outputs."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "output", "data")
FIG = os.path.join(ROOT, "output", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})


def fig_e1():
    df = pd.read_csv(os.path.join(DATA, "e1_benchmark.csv"))
    labels = {"no_reset": "No reset", "endogenous": "Endogenous",
              "exogenous_rho0.5": "Exogenous ($\\rho=0.5$)"}
    order = ["no_reset", "endogenous", "exogenous_rho0.5"]
    x = np.arange(len(order))
    prices = [df[df.model == m].price.iloc[0] for m in order]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(x, prices, color=["#7f8c8d", "#2c3e50", "#e74c3c"], width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in order])
    ax.set_ylabel("Bond value (per 100 face)")
    ax.set_title("Convertible bond value under three reset regimes")
    for b, p in zip(bars, prices):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.8, f"{p:.1f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(105, 135)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e1_benchmark.png"), dpi=200)
    plt.close(fig)


def fig_e2():
    df = pd.read_csv(os.path.join(DATA, "e2_reset_boundary.csv"))
    pivot_price = df.pivot(index="C0", columns="S0", values="price")
    pivot_freq = df.pivot(index="C0", columns="S0", values="reset_freq")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    im1 = axes[0].imshow(pivot_price.values, aspect="auto", origin="lower",
                         cmap="viridis")
    axes[0].set_xticks(range(len(pivot_price.columns)))
    axes[0].set_xticklabels([f"{int(c)}" for c in pivot_price.columns])
    axes[0].set_yticks(range(len(pivot_price.index)))
    axes[0].set_yticklabels([f"{int(c)}" for c in pivot_price.index])
    axes[0].set_xlabel("Initial stock price $S_0$")
    axes[0].set_ylabel("Initial capital $C_0$")
    axes[0].set_title("Bond value")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(pivot_freq.values, aspect="auto", origin="lower",
                         cmap="plasma", vmin=0, vmax=2)
    axes[1].set_xticks(range(len(pivot_freq.columns)))
    axes[1].set_xticklabels([f"{int(c)}" for c in pivot_freq.columns])
    axes[1].set_yticks(range(len(pivot_freq.index)))
    axes[1].set_yticklabels([f"{int(c)}" for c in pivot_freq.index])
    axes[1].set_xlabel("Initial stock price $S_0$")
    axes[1].set_ylabel("Initial capital $C_0$")
    axes[1].set_title("Average number of resets per path")
    fig.colorbar(im2, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e2_boundary.png"), dpi=200)
    plt.close(fig)


def fig_e3():
    df = pd.read_csv(os.path.join(DATA, "e3_sensitivity.csv"))
    params = ["eta", "beta", "sigma_c", "C_reset"]
    names = {"eta": "$\\eta$ (distress drift)",
             "beta": "$\\beta$ (liquidity vol)",
             "sigma_c": "$\\sigma_c$ (capital vol)",
             "C_reset": "$C_{reset}$ (reset threshold)"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, prm in zip(axes.ravel(), params):
        sub = df[df.param == prm]
        ax.plot(sub.value, sub.price, "o-", color="#2c3e50")
        ax.set_xlabel(names[prm])
        ax.set_ylabel("Bond value")
        ax.set_title(f"Sensitivity to {names[prm]}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e3_sensitivity.png"), dpi=200)
    plt.close(fig)


def fig_e4():
    df = pd.read_csv(os.path.join(DATA, "e4_convergence.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(df.N, df.price, yerr=df["std"], fmt="o-", color="#2c3e50",
                capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths $N$")
    ax.set_ylabel("Bond value")
    ax.set_title("LSM estimator convergence")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e4_convergence.png"), dpi=200)
    plt.close(fig)


def fig_e5():
    df = pd.read_csv(os.path.join(DATA, "e5_robustness_threshold.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(df.C_reset, df.price, yerr=df["se"], fmt="o-", color="#2c3e50",
                capsize=3)
    ax.axvline(150.0, color="#e74c3c", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Strategic revision threshold $C_{reset}$")
    ax.set_ylabel("Bond value")
    ax.set_title("Bond value against the strategic threshold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e5_threshold.png"), dpi=200)
    plt.close(fig)


def main():
    fig_e1()
    fig_e2()
    fig_e3()
    fig_e4()
    fig_e5()
    print("[figures] done")


if __name__ == "__main__":
    main()

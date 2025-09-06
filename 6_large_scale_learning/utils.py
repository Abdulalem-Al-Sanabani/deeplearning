import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def sanity_check_add_training_tokens(df):
    values = df["Training Tokens"][:5].values
    ref_values = np.array(
        [
            237262970.49876866,
            494690375.7031911,
            589545586.2098124,
            671155986.9785657,
            774127437.8382856,
        ]
    )
    assert np.allclose(values, ref_values), f"Expected {ref_values} but got {values}"
    print("Sanity check passed!")


def preprocess_df(df):
    assert "Training Tokens" in df.columns, "Training Tokens column is missing."

    df = df[["Model Size", "Training Tokens", "Training FLOP", "loss"]].dropna()
    df["d_n_ratio"] = df["Training Tokens"] / df["Model Size"]

    # drop outliers
    df = df.drop(df["d_n_ratio"].sort_values().head(5).index)
    df = df.drop(df["d_n_ratio"].where(lambda x: x < 1).dropna().index)
    return df


def _plot_vertical_density(ax, x, y):
    assert x.shape == y.shape and x.ndim == 1
    n_samples = 200
    y_sample = np.linspace(y.min(), y.max(), n_samples)
    x_sample = gaussian_kde(y)(y_sample)

    # determine x position
    x_pos = x.max() + 0.05 * (x.max() - x.min())

    # determine density amplitude
    x_scale = 0.8 * (x.max() - x.min()) / x_sample.max()

    # plot density
    ax.fill_betweenx(y_sample, x_pos, x_pos + x_sample * x_scale, alpha=0.5)
    ax.plot(x_pos + x_sample * x_scale, y_sample, color="black", linewidth=1)
    ax.plot([x_pos] * n_samples, y_sample, color="black", linewidth=1)


def plot_residuals(params, df, title):
    N = df["Model Size"].values
    D = df["Training Tokens"].values
    losses = df["loss"].values

    A = params["A"]
    B = params["B"]
    E = params["E"]
    alpha = params["alpha"]
    beta = params["beta"]

    residuals = losses - (E + A / (N**alpha) + B / (D**beta))

    # Create scatter plot of residuals
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(losses, residuals, alpha=0.35, color="blue")
    _plot_vertical_density(ax, losses, residuals)
    ax.set_xlabel("Losses")
    ax.set_ylabel("Residuals")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=3)
    ax.set_title(title)
    plt.show()


PADDING = 0.1


def _adjust_subplot(ax, x, y, xlim, ylim):
    """Adjusts the subplot configurations and return the bound of values."""
    # Get a tuple of value range for each axis, letting the minimum of a seed range the lowest and padding the all data points
    limits_by_k = {}
    for k in [x, y]:
        # Converting to `float` in case of int dtype (`np.log` cannot intake astronomically large integers)
        max_value = {x: xlim[1], y: ylim[1]}[k] / 1
        min_value = {x: xlim[0], y: ylim[0]}[k] / 1
        log_range = np.log(max_value) - np.log(min_value)
        limits_by_k[k] = (
            min_value * np.exp(-log_range * PADDING),
            max_value * np.exp(log_range * PADDING),
        )

    # Apply the bounds
    xlim, ylim = limits_by_k[x], limits_by_k[y]
    if np.isnan(xlim).any():
        raise ValueError(f"NaN value(s) found for the x axis {x} bound: {xlim}")
    if np.isnan(ylim).any():
        raise ValueError(f"NaN value(s) found for the x axis {y} bound: {ylim}")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xscale("log")
    ax.set_yscale("log")
    to_label = {"C": "FLOPs", "D": "Samples", "N": "Parameters"}
    ax.set_xlabel(f"{to_label[x]} (${x}$)")
    ax.set_ylabel(f"{to_label[y]} (${y}$)")

    return xlim, ylim


# possible plots: N vs D, N vs C, D vs C
def _get_isoloss_values(loss_values, params, N=None, D=None):
    # unsafe but works
    A, B, E, alpha, beta = params.values()
    assert N is not None or D is not None, "Either N or D must be provided."
    if N is not None:
        D = np.power(
            B / (loss_values - (E + np.exp(np.log(A) - alpha * np.log(N)))), 1 / beta
        )
    if D is not None:
        N = np.power(
            A / (loss_values - (E + np.exp(np.log(B) - beta * np.log(D)))), 1 / alpha
        )
    C = 6 * N * D
    return {"N": N, "D": D, "C": C}


def _plot_loss_gradient(ax, xaxis, yaxis, params, iso_losses, y_lim):
    """Helper method to plot the loss gradient."""
    y_min, y_max = y_lim
    log_ymin = np.log10(y_min)
    log_ymax = np.log10(y_max)
    log_ymin -= (log_ymax - log_ymin) * PADDING
    log_ymax += (log_ymax - log_ymin) * PADDING

    y_values = np.logspace(log_ymin, log_ymax, 1000, dtype=np.double)
    assert not np.isnan(y_values).sum(), (
        f"{100*np.isnan(y_values).mean()}% NaN:",
        y_values,
    )

    values = {yaxis: y_values}
    for j, L in enumerate(iso_losses):
        ndc = _get_isoloss_values(L, params, **values)

        x_values = ndc[xaxis]
        ax.plot(
            x_values,
            y_values,
            zorder=0,
            c=plt.get_cmap("magma")(j / len(iso_losses)),
            linewidth=1.5,
        )


def _get_optimal_values(params, C):
    A, B, E, alpha, beta = params.values()
    a, b = beta / (alpha + beta), alpha / (alpha + beta)
    G = np.power((alpha * A) / (beta * B), 1 / (alpha + beta))
    N_opt = G * (C / 6) ** a
    D_opt = (1 / G) * (C / 6) ** b
    return {"C": C, "N": N_opt, "D": D_opt}


def _plot_predicted_optimal(ax, xaxis, yaxis, params, C):
    """Helper method to plot the predicted optimal point."""
    C_range = np.logspace(15, 25, 3)
    ndc = _get_optimal_values(params, C_range)
    ax.plot(ndc[xaxis], ndc[yaxis], color="blue", zorder=-1)
    ndc = _get_optimal_values(params, C)
    ax.plot(
        ndc[xaxis],
        ndc[yaxis],
        marker="*",
        markersize=20,
        color="blue",
        markeredgecolor="black",
        zorder=10,
    )
    ax.hlines(
        ndc[yaxis],
        0,
        ndc[xaxis],
        color="black",
        linestyle="--",
        linewidth=1,
        zorder=-1,
    )
    ax.vlines(
        ndc[xaxis],
        0,
        ndc[yaxis],
        color="black",
        linestyle="--",
        linewidth=1,
        zorder=-1,
    )

    # ax.text(ax.get_xlim()[0], ndc[yaxis], f"{ndc[yaxis]:.2e}", fontsize=12, ha="left", va="bottom")
    # ax.text(ndc[xaxis], ax.get_ylim()[0], f"{ndc[xaxis]:.2e}", fontsize=12, ha="right", va="bottom")


def model_scaling_plot(params, df, C=5.76e23):
    values = {
        "C": df["Training FLOP"],
        "N": df["Model Size"],
        "D": df["Training Tokens"],
    }
    limits = {
        "C": (df["Training FLOP"].min(), 1e24),
        "N": (df["Model Size"].min(), 1e11),
        "D": (df["Training Tokens"].min(), 1e13),
    }
    # compute suitable iso-loss values
    losses = df["loss"].values
    loss_min, loss_max = np.min(losses), np.max(losses)
    margin = 0.15
    loss_range = loss_max - loss_min
    iso_losses = np.linspace(
        loss_min - loss_range * margin, loss_max + loss_range * margin, 24
    )
    ndc = _get_optimal_values(params, C)
    print("Optimal for C={C:.2e}: N={N:.2e}, D={D:.2e}".format(**ndc))

    for xaxis, yaxis in [("C", "N"), ("C", "D"), ("N", "D")]:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        _adjust_subplot(ax, xaxis, yaxis, limits[xaxis], limits[yaxis])
        sc = ax.scatter(
            values[xaxis],
            values[yaxis],
            c=df["loss"],
            edgecolor="black",
            linewidths=0.5,
            vmin=loss_min,
            vmax=loss_max,
        )
        fig.colorbar(sc, label="Loss")

        _plot_loss_gradient(ax, xaxis, yaxis, params, iso_losses, limits[yaxis])

        _plot_predicted_optimal(ax, xaxis, yaxis, params, C)
        fig.show()

from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ################################################
#          From Telco notebook
# ################################################

# Kolmogorov Smirnov
def plot_ks(model, X_list, y_list, name_list):
    n = len(X_list)

    eval_points_list = []
    positive_cdf_interp_list = []
    negative_cdf_interp_list = []
    ks_stat_position_list = []
    ks_stat_list = []

    # Calculation
    for i, (X, y, name) in enumerate(zip(X_list, y_list, name_list)):
        true_labels = y
        predicted_proba = model.predict_proba(X)[:, 1]

        data = pd.DataFrame({
            'true_labels': true_labels,
            'predicted_proba': predicted_proba
        })

        # Separate the probabilities into two groups based on true labels
        positive_proba = data[data['true_labels'] == 1]['predicted_proba']
        negative_proba = data[data['true_labels'] == 0]['predicted_proba']

        # Compute KS statistic and p-value using ks_2samp
        ks_stat, p_value = ks_2samp(positive_proba, negative_proba)

        # Compute cumulative distributions
        positive_cdf = np.sort(positive_proba)
        positive_cdf_values = np.arange(1, len(positive_cdf) + 1) / len(positive_cdf)

        negative_cdf = np.sort(negative_proba)
        negative_cdf_values = np.arange(1, len(negative_cdf) + 1) / len(negative_cdf)

        # Interpolation for a common set of points
        eval_points = np.sort(np.unique(np.concatenate([positive_cdf, negative_cdf])))
        positive_cdf_interp = np.interp(eval_points, positive_cdf, positive_cdf_values, left=0, right=1)
        negative_cdf_interp = np.interp(eval_points, negative_cdf, negative_cdf_values, left=0, right=1)

        # Calculate the KS statistic
        ks_stat = np.max(np.abs(positive_cdf_interp - negative_cdf_interp))

        ks_stat_position = eval_points[np.argmax(np.abs(positive_cdf_interp - negative_cdf_interp))]

        eval_points_list.append(eval_points)
        positive_cdf_interp_list.append(positive_cdf_interp)
        negative_cdf_interp_list.append(negative_cdf_interp)
        ks_stat_position_list.append(ks_stat_position)
        ks_stat_list.append(ks_stat)

    # Plot cumulative distributions
    fig, axes = plt.subplots(ncols=n, figsize=(5*n, 4), squeeze=False)
    for i, (name, eval_points, positive_cdf_interp, negative_cdf_interp, ks_stat_position, ks_stat, ax) in enumerate(zip(name_list, eval_points_list, positive_cdf_interp_list, negative_cdf_interp_list, ks_stat_position_list, ks_stat_list, axes.flatten())):
        ax.plot(eval_points, positive_cdf_interp, label='Positive CDF', color='blue')
        ax.plot(eval_points, negative_cdf_interp, label='Negative CDF', color='red')
        ax.axvline(x=ks_stat_position, color='green', linestyle='--', label=f'KS Statistic = {ks_stat:.4f}')
        ax.set_title(f'Kolmogorov-Smirnov Cumulative Distribution\nfor {name} data set.', fontsize=10)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Cumulative Distribution')
        ax.legend()
        ax.grid()

    plt.close()
    return fig, axes
    # return eval_points_list, positive_cdf_interp_list, negative_cdf_interp_list, ks_stat_position_list, ks_stat_list


def calculate_ks(model, X, y):
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic and p-value for a given model and dataset.

    Parameters
    ----------
    model : object
        A model with a predict_prob method that returns predicted probabilities.
    X : array-like or DataFrame
        Feature matrix.
    y : array-like or Series
        True labels (0 or 1).

    Returns
    -------
    ks_stat : float
        KS statistic value.
    p_value : float
        Two-tailed p-value associated with the KS statistic.
    """
    # Get predicted probabilities
    predicted_proba = model.predict_prob(X)

    # Separate probabilities by class
    positive_proba = predicted_proba[y == 1]
    negative_proba = predicted_proba[y == 0]

    # Calculate the KS statistic and p-value
    ks_stat, p_value = ks_2samp(positive_proba, negative_proba)

    return ks_stat, p_value


# ################################################
#          From Data Science studies repo
# ################################################


def calculate_ks(y, y_prob):
    if isinstance(y, pd.DataFrame):
        y = y.values

    if isinstance(y, np.ndarray):
        if len(y.shape) > 1:
            y = y.flatten()

    true_labels = y
    predicted_proba = y_prob

    data = pd.DataFrame({
        'true_labels': true_labels,
        'predicted_proba': predicted_proba
    })

    # Separate the probabilities into two groups based on true labels
    positive_proba = data[data['true_labels'] == 1]['predicted_proba']
    negative_proba = data[data['true_labels'] == 0]['predicted_proba']

    # Compute KS statistic and p-value using ks_2samp
    ks_stat, p_value = ks_2samp(positive_proba, negative_proba)

    return ks_stat, p_value

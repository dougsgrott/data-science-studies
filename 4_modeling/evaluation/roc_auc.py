from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import plotly.graph_objects as go
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt


def plotly_roc_auc(dataset_dict):

    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    for set_name in dataset_dict:
        y_ = dataset_dict[set_name]['y']
        y_prob = dataset_dict[set_name]['y_prob'][:, 1]

        fpr, tpr, thresholds = roc_curve(y_, y_prob)
        auc_score = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{set_name} (AUC={auc_score:.3f})", mode='lines'))

    fig.update_layout(
        title=f"ROC AUC Curve",
        title_x=0.5,
        xaxis=dict(
            title=dict(text='False Positive Rate'),
            constrain='domain',
            title_font_size=20,
        ),
        yaxis=dict(
            title=dict(text='True Positive Rate'),
            scaleanchor='x',
            scaleratio=1,
            title_font_size=20,
        ),
        width=700, height=500
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig


def plotly_pr_auc(dataset_dict):
    fig = go.Figure()
    
    # Add horizontal baseline for no-skill classifier
    for set_name in dataset_dict:
        y_ = dataset_dict[set_name]['y']
        base_rate = y_.mean()
        fig.add_shape(
            type='line', line=dict(dash='dash', color='gray'),
            x0=0, x1=1, y0=base_rate, y1=base_rate,
            name='Baseline'
        )
        break  # Add baseline once

    # Plot PR curves for each dataset
    for set_name in dataset_dict:
        y_true = dataset_dict[set_name]['y']
        y_prob = dataset_dict[set_name]['y_prob'][:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)

        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f"{set_name} (AP={ap_score:.3f})"
        ))

    # Layout
    fig.update_layout(
        title="Precision-Recall Curve",
        title_x=0.5,
        xaxis=dict(
            title=dict(text='Recall'),
            constrain='domain',
            title_font_size=20,
        ),
        yaxis=dict(
            title=dict(text='Precision'),
            title_font_size=20,
            scaleratio=1,
        ),
        width=700, height=500,
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig


def plot_roc_auc(classifier, dataset_dict, **kwargs):
    fig, axes = plt.subplots(figsize=(5,5))
    for set_name in dataset_dict:
        x_ = dataset_dict[set_name]['x']
        y_ = dataset_dict[set_name]['y']
        roc = RocCurveDisplay.from_estimator(classifier, x_, y_, ax=axes)
        roc.ax_.set_title('ROC Plot')
        # if show: plt.show()
    plt.close()
    return roc.figure_


def create_roc_auc_plot(classifier, dataset_dict, **kwargs):
    interactive = kwargs.pop('interactive', True)
    show = kwargs.pop('show', True)

    if interactive == False:
        fig = plot_roc_auc(classifier, dataset_dict, **kwargs)
        return fig
    fig = plotly_roc_auc(classifier, dataset_dict, **kwargs)
    # if show: fig.show()
    return fig

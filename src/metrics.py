from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# report claim frequency by sum of claims / sum of exposure
def calculate_claim_frequency(learn_data, test_data):
    cf_l = round(learn_data['ClaimNb'].sum() / learn_data['Exposure'].sum() * 100, 1)
    cf_t = round(test_data['ClaimNb'].sum() / test_data['Exposure'].sum() * 100, 1)
    print(f"Claim frequency for learn data: {cf_l} %")
    print(f"Claim frequency for test data: {cf_t} %")
    # Calculate claim frequency for learn data
    return None


def plot_obs_pred(
    df,
    feature,
    weight,
    observed,
    predicted,
    y_label=None,
    title=None,
    ax=None,
    fill_legend=False,
):

    # aggregate observed and predicted variables by feature level
    df_ = df.loc[:, [feature, weight]].copy()
    df_["observed"] = df[observed] * df[weight]
    df_["predicted"] = predicted * df[weight]
    df_ = (
        df_.groupby([feature])[[weight, "observed", "predicted"]]
        .sum()
        .assign(observed=lambda x: x["observed"] / x[weight])
        .assign(predicted=lambda x: x["predicted"] / x[weight])
    )

    ax = df_.loc[:, ["observed", "predicted"]].plot(style=".", ax=ax)
    y_max = df_.loc[:, ["observed", "predicted"]].values.max() * 0.8
    p2 = ax.fill_between(
        df_.index,
        0,
        y_max * df_[weight] / df_[weight].values.max(),
        color="g",
        alpha=0.1,
    )
    if fill_legend:
        ax.legend([p2], ["{} distribution".format(feature)])
    ax.set(
        ylabel=y_label if y_label is not None else None,
        title=title if title is not None else "Train: Observed vs Predicted",
    )

def score_estimator(trained_model, X_train, X_test, df_train, df_test, target, weights):    
    metrics = [
        ("square R score", r2_score),  # Use default scorer if it exists
        ("mean abs. error", mean_absolute_error),
        ("mean squared error", mean_squared_error),
    ]
    res = []
    
    for subset_label, X, df in [("train", X_train, df_train), ("test", X_test, df_test)]:
        y, _weights = df[target], df[weights]
        for score_label, metric in metrics:
            y_pred = trained_model.predict(X)
            score = metric(y, y_pred, sample_weight=_weights)
            res.append({"subset": subset_label, "metric": score_label, "score": score})
    res = (
        pd.DataFrame(res)
        .set_index(["metric", "subset"])
        .score.unstack(-1)
        .round(4)
        .loc[:, ["train", "test"]]
    )
    return res
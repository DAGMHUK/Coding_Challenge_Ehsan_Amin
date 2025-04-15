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


def score_estimator(trained_model, X_train, X_test, df_train, df_test, target, weights):    
    metrics = [
        ("mean abs. error", mean_absolute_error),
        ("R-squared score", r2_score),  # Use default scorer if it exists
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
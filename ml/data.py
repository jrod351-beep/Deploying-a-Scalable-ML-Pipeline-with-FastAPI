
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=None, label=None, training=True, encoder=None, lb=None
):
    """Process data for the machine learning pipeline."""
    if categorical_features is None:
        categorical_features = []

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features)

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        if label is not None:
            y = lb.transform(y.values).ravel()

    X = np.concatenate([X_continuous.to_numpy(), X_categorical], axis=1)
    return X, y, encoder, lb


def apply_label(inference):
    """Convert a binary prediction into its string label."""
    if inference[0] == 1:
        return ">50K"
    elif inference[0] == 0:
        return "<=50K"

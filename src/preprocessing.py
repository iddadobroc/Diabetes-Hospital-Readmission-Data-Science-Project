"""The preprocessing of each model family, as scikit-learn transformers.

Everything here LEARNS from the data (frequent levels, means, standard deviations), so
it is only ever fitted inside a `Pipeline`, on the fitting rows of each split or fold.
It rebuilds the three variants of Luis Alcaide's `fit_preprocessor`
(`LA_Models/01_preprocessing_and_models.ipynb`, Stage 3):

| kind      | for                                    | numbers      | categories    |
|-----------|----------------------------------------|--------------|---------------|
| `linear`  | logistic regression, neural networks   | standardised | one-hot       |
| `onehot`  | random forest, XGBoost, LightGBM       | as they are  | one-hot       |
| `ordinal` | histogram gradient boosting            | as they are  | integer codes |
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC

# Rare levels (D-009): at most the TOP_N most frequent levels with MIN_ROWS rows
TOP_N, MIN_ROWS = 10, 100


def _encoder(kind, min_frequency, max_categories):
    """One categorical encoder; unseen levels go to the infrequent level or NaN."""
    if kind == "ordinal":
        return OrdinalEncoder(
            min_frequency=min_frequency,
            max_categories=max_categories,
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )
    return OneHotEncoder(
        min_frequency=min_frequency,
        max_categories=max_categories,
        handle_unknown="infrequent_if_exist" if min_frequency else "ignore",
        sparse_output=False,
    )


def make_preprocessor(
    kind,
    numeric=NUMERIC,
    categorical=CATEGORICAL,
    all_levels=("gender",),
    no_top_n=(),
):
    """The preprocessing of one model family, as an unfitted ColumnTransformer.

    Args:
        kind (str): "linear", "onehot" or "ordinal" (see the module table).
        numeric (list[str]): The numeric feature columns.
        categorical (list[str]): The categorical feature columns.
        all_levels (tuple[str]): Categorical columns whose levels are all kept,
            with no rare-level grouping (`gender`, as the brief asks).
        no_top_n (tuple[str]): Categorical columns where levels with fewer than
            `MIN_ROWS` rows are grouped, but the number of levels is not capped
            (for a column already grouped by hand, such as `medical_specialty`
            mapped to clinical areas).

    Returns:
        ColumnTransformer: Learns nothing until it is fitted. Its output is a
        DataFrame that keeps the column names.
    """
    capped = [c for c in categorical if c not in all_levels and c not in no_top_n]
    blocks = [
        ("num", StandardScaler() if kind == "linear" else "passthrough", numeric),
        ("cat", _encoder(kind, MIN_ROWS, TOP_N + 1), capped),
        ("cat_uncapped", _encoder(kind, MIN_ROWS, None), list(no_top_n)),
        ("cat_all_levels", _encoder(kind, None, None), list(all_levels)),
    ]
    return ColumnTransformer(
        [block for block in blocks if len(block[2])],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_pipeline(kind, model, **columns):
    """Preprocessing + model in one object, fitted together on the fitting rows.

    Args:
        kind (str): The preprocessing variant, passed to `make_preprocessor`.
        model: An unfitted scikit-learn compatible classifier.
        **columns: Optional `numeric`, `categorical`, `all_levels`, `no_top_n`,
            passed to `make_preprocessor`.

    Returns:
        Pipeline: The steps "prep" and "model".
    """
    return Pipeline([("prep", make_preprocessor(kind, **columns)), ("model", model)])

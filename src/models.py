"""Scikit-learn wrappers for models that have no ready-made classifier interface.

Both classes follow the scikit-learn estimator contract (`fit`, `predict_proba`,
`get_params`), so they can sit inside a `Pipeline` and be cloned, cross-validated and
tuned exactly like the other models, with the preprocessing refitted in every fold.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight

HORIZON_DAYS = 30  # the readmission window of the target


class XGBAFTClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost accelerated failure time (AFT) model, read as a 30-day classifier.

    An AFT model describes the time to the event: log(T) = f(x) + sigma * Z, where Z
    follows `distribution`. The dataset has no readmission dates, only "<30", ">30"
    and "NO", so the time is interval-censored:

    - readmitted in less than 30 days (`y = 1`): T in [0, 30];
    - otherwise (`y = 0`): T in [30, +inf), right-censored at 30 days.

    `predict_proba` returns P(T <= 30) = F((log 30 - f(x)) / sigma), the probability of
    a readmission within the horizon. With only this information the AFT model is a
    binary classifier with a different link function (normal, logistic or extreme
    value) and its own way of weighting the errors.

    Boosting starts from a time of 30 days (`base_score`), the boundary between the two
    groups. XGBoost's default start (0.5 days) makes the training diverge for small
    `sigma` (every predicted log-time ends near -3000, the same for all patients).

    Args:
        distribution (str): "normal", "logistic" or "extreme".
        sigma (float): Scale of the distribution (`aft_loss_distribution_scale`).
        n_estimators (int): Number of boosting rounds (fixed, no early stopping).
        learning_rate, max_depth, min_child_weight, subsample, colsample_bytree,
        reg_lambda, max_delta_step: As in `xgboost.XGBClassifier`; `max_delta_step`
        caps each update of the log-time, a second guard against divergence.
        balanced (bool): Weight the classes inversely to their frequency.
        random_state (int): Seed.
        n_jobs (int): Threads used by XGBoost.
    """

    def __init__(
        self,
        distribution="normal",
        sigma=1.0,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        max_delta_step=0.7,
        balanced=True,
        random_state=0,
        n_jobs=-1,
    ):
        self.distribution = distribution
        self.sigma = sigma
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.max_delta_step = max_delta_step
        self.balanced = balanced
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the AFT model on interval-censored times built from the binary target.

        Args:
            X (array-like): Numeric features (already preprocessed).
            y (array-like): 1 = readmitted in less than 30 days, 0 = not.

        Returns:
            XGBAFTClassifier: self.
        """
        import xgboost as xgb

        y = np.asarray(y).astype(int)
        lower = np.where(y == 1, 0.0, HORIZON_DAYS)
        upper = np.where(y == 1, HORIZON_DAYS, np.inf)
        weight = compute_sample_weight("balanced", y) if self.balanced else None
        data = xgb.DMatrix(np.asarray(X, dtype=float), weight=weight)
        data.set_float_info("label_lower_bound", lower)
        data.set_float_info("label_upper_bound", upper)
        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": self.distribution,
            "aft_loss_distribution_scale": self.sigma,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "lambda": self.reg_lambda,
            "max_delta_step": self.max_delta_step,
            "base_score": float(HORIZON_DAYS),
            "tree_method": "hist",
            "seed": self.random_state,
            "nthread": self.n_jobs,
        }
        self.booster_ = xgb.train(params, data, num_boost_round=self.n_estimators)
        log_time = self.booster_.predict(data, output_margin=True)
        if not np.isfinite(log_time).all() or np.abs(np.median(log_time)) > 50:
            warnings.warn(
                "XGBAFTClassifier: the training diverged (predicted log-times "
                f"around {np.median(log_time):.0f}); its predictions are not usable."
            )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """P(not readmitted in 30 days), P(readmitted in 30 days) for each row."""
        import xgboost as xgb

        data = xgb.DMatrix(np.asarray(X, dtype=float))
        log_time = self.booster_.predict(data, output_margin=True)
        # Clipped so that exp() cannot overflow; |z| = 30 is already p = 0 or 1
        z = np.clip((np.log(HORIZON_DAYS) - log_time) / self.sigma, -30, 30)
        if self.distribution == "normal":
            p = norm.cdf(z)
        elif self.distribution == "logistic":
            p = 1.0 / (1.0 + np.exp(-z))
        else:  # "extreme": minimum extreme value (Gumbel) distribution
            p = 1.0 - np.exp(-np.exp(z))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """1 if the probability of a readmission within 30 days is 0.5 or more."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class CatBoostNativeClassifier(ClassifierMixin, BaseEstimator):
    """CatBoost reading the text columns of a DataFrame as categorical features.

    `catboost.CatBoostClassifier` cannot be cloned by scikit-learn when `cat_features`
    is given (it changes that parameter), and cloning is what every cross-validation
    and search does. This wrapper finds the categorical columns itself, in `fit` (every
    non-numeric column), so it clones like any scikit-learn model.

    Args:
        iterations, learning_rate, depth, l2_leaf_reg: As in `CatBoostClassifier`.
        balanced (bool): `auto_class_weights="Balanced"`.
        random_state (int): Seed.
        thread_count (int): Threads used by CatBoost (-1 = all).
    """

    def __init__(
        self,
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        balanced=True,
        random_state=0,
        thread_count=-1,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.balanced = balanced
        self.random_state = random_state
        self.thread_count = thread_count

    def fit(self, X, y):
        """Fit CatBoost; the text columns of `X` are its categorical features.

        Args:
            X (pandas.DataFrame): Raw features, categories as text.
            y (array-like): 1 = readmitted in less than 30 days, 0 = not.

        Returns:
            CatBoostNativeClassifier: self.
        """
        from catboost import CatBoostClassifier

        self.cat_features_ = [
            c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
        ]
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            auto_class_weights="Balanced" if self.balanced else None,
            cat_features=self.cat_features_,
            random_seed=self.random_state,
            thread_count=self.thread_count,
            verbose=0,
            allow_writing_files=False,  # no catboost_info/ folder
        )
        self.model_.fit(X, np.asarray(y).astype(int))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """P(not readmitted in 30 days), P(readmitted in 30 days) for each row."""
        return self.model_.predict_proba(X)

    def predict(self, X):
        """1 if the probability of a readmission within 30 days is 0.5 or more."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class KerasMLPClassifier(ClassifierMixin, BaseEstimator):
    """A Keras multi-layer perceptron for tabular data, as a scikit-learn classifier.

    It expects numbers that are already scaled and one-hot encoded (the `linear`
    preprocessing). It trains for a FIXED number of epochs: Keras' `validation_split`
    would hold out the last rows by encounter, not by patient, so early stopping on
    it would mix patients (leakage). The number of epochs is tuned by grouped
    cross-validation instead, like any other hyperparameter.

    Args:
        hidden (tuple[int]): Neurons of each hidden layer.
        dropout (float): Dropout rate after each hidden layer.
        l2 (float): L2 penalty on the weights.
        learning_rate (float): Adam learning rate.
        epochs (int): Number of passes over the training rows.
        batch_size (int): Rows per gradient step.
        balanced (bool): Weight the classes inversely to their frequency.
        random_state (int): Seed of Keras, NumPy and Python's `random`.
    """

    def __init__(
        self,
        hidden=(64, 32),
        dropout=0.3,
        l2=1e-4,
        learning_rate=1e-3,
        epochs=10,
        batch_size=512,
        balanced=True,
        random_state=0,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.balanced = balanced
        self.random_state = random_state

    def fit(self, X, y):
        """Build the network and train it for `epochs` epochs.

        Args:
            X (array-like): Scaled, one-hot encoded features.
            y (array-like): 1 = readmitted in less than 30 days, 0 = not.

        Returns:
            KerasMLPClassifier: self.
        """
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras

        keras.utils.set_random_seed(self.random_state)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y).astype("float32")
        layers = [keras.Input(shape=(X.shape[1],))]
        for units in self.hidden:
            layers.append(
                keras.layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(self.l2),
                )
            )
            layers.append(keras.layers.Dropout(self.dropout))
        layers.append(keras.layers.Dense(1, activation="sigmoid"))
        self.model_ = keras.Sequential(layers)
        self.model_.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
        )
        class_weight = None
        if self.balanced:
            positive = y.mean()
            class_weight = {0: 0.5 / (1 - positive), 1: 0.5 / positive}
        self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            shuffle=True,
            verbose=0,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """P(not readmitted in 30 days), P(readmitted in 30 days) for each row."""
        p = self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """1 if the probability of a readmission within 30 days is 0.5 or more."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

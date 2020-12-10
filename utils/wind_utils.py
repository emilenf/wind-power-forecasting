import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from utils.wind_configs import Config, MetaLogger


def read_data(data_folder_path, file_name):
    return pd.read_csv(os.path.join(data_folder_path, file_name))


def day_sin_cycle(t, k):
    sin_day = np.sin(k * t.hour * 2 * np.pi / 24)
    return sin_day


def day_cos_cycle(t, k):
    cos_day = np.cos(k * t.hour * 2 * np.pi / 24)
    return cos_day


def year_sin_cycle(t, k):
    first_day_of_current_year = datetime(t.year, 1, 1, 0, 0, 0)
    time_diff = t - first_day_of_current_year
    time_diff_hours = time_diff.days * 24
    sin_year = np.sin(k * time_diff_hours * 2 * np.pi / 8760)
    return sin_year


def year_cos_cycle(t, k):
    first_day_of_current_year = datetime(t.year, 1, 1, 0, 0, 0)
    time_diff = t - first_day_of_current_year
    time_diff_hours = time_diff.days * 24
    cos_year = np.cos(k * time_diff_hours * 2 * np.pi / 8760)
    return cos_year


def fill_run_nans_w_previous_run(data, run, model, var):
    if model in ["NWP1", "NWP3"]:
        data[run] = (
            data[run]
            .fillna(data[f"{model}_12h_D_{var}"])
            .fillna(data[f"{model}_06h_D_{var}"])
            .fillna(data[f"{model}_00h_D_{var}"])
            .fillna(data[f"{model}_18h_D-1_{var}"])
            .fillna(data[f"{model}_12h_D-1_{var}"])
            .fillna(data[f"{model}_06h_D-1_{var}"])
            .fillna(data[f"{model}_00h_D-1_{var}"])
            .fillna(data[f"{model}_18h_D-2_{var}"])
            .fillna(data[f"{model}_12h_D-2_{var}"])
            .fillna(data[f"{model}_06h_D-2_{var}"])
            .fillna(data[f"{model}_00h_D-2_{var}"])
        )

    else:
        data[run] = (
            data[run]
            .fillna(data[f"{model}_00h_D_{var}"])
            .fillna(data[f"{model}_12h_D-1_{var}"])
            .fillna(data[f"{model}_00h_D-1_{var}"])
            .fillna(data[f"{model}_12h_D-2_{var}"])
            .fillna(data[f"{model}_00h_D-2_{var}"])
        )

    return data[run]


def wind_components_to_speed(u, v):
    wspeed = np.sqrt(u ** 2 + v ** 2)
    return wspeed


def wind_components_to_dir(u, v):
    wdir = (270 - np.rad2deg(np.arctan2(v, u))) % 360
    return wdir


class DataLoader:
    def __init__(self,):
        self.logger = MetaLogger.setup_logging(__name__)
        MetaLogger.log_message(self.logger, f"Loading datasets")
        self.__load_data()

    def __load_data(self) -> None:
        self.X_train = read_data(Config.data_folder_path, Config.x_train_file).drop(
            ["ID"], axis=1
        )
        MetaLogger.log_message(self.logger, "X_train loaded")
        self.y_train = read_data(Config.data_folder_path, Config.y_train_file)
        MetaLogger.log_message(self.logger, "y_train loaded")
        self.X_train["Production"] = self.y_train["Production"]
        self.X_test = read_data(Config.data_folder_path, Config.x_test_file)
        MetaLogger.log_message(self.logger, "X_test loaded")
        self.complementary_data = pd.read_csv(
            os.path.join(Config.data_folder_path, Config.complementary_data_file),
            sep=";",
            encoding="unicode_escape",
        )
        MetaLogger.log_message(self.logger, "Complementary_data loaded")


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, k, n_lags):
        self.k = k
        self.n_lags = n_lags

    def _get_cyclical_variables(self, X):
        X["sin_day"] = X.index.to_series().apply(lambda x: day_sin_cycle(x, k=self.k))
        X["cos_day"] = X.index.to_series().apply(lambda x: day_cos_cycle(x, k=self.k))
        X["sin_year"] = X.index.to_series().apply(lambda x: year_sin_cycle(x, k=self.k))
        X["cos_year"] = X.index.to_series().apply(lambda x: year_cos_cycle(x, k=self.k))
        return X

    def _get_lagged_feature(self, X, feature, n_lags):
        for obs in range(1, n_lags):
            X[f"{feature}_T-{obs}"] = X[feature].shift(obs)
        return X

    def _fill_all_runs(self, data):
        for var in Config.variables_npw13:

            data[f"SELECTED_NWP1_{var}"] = fill_run_nans_w_previous_run(
                data=data, run=f"NWP1_18h_D_{var}", model="NWP1", var=var
            )
            data[f"SELECTED_NWP3_{var}"] = fill_run_nans_w_previous_run(
                data=data, run=f"NWP3_18h_D_{var}", model="NWP3", var=var
            )

        for var in Config.variables_npw2:
            data[f"SELECTED_NWP2_{var}"] = fill_run_nans_w_previous_run(
                data=data, run=f"NWP2_12h_D_{var}", model="NWP2", var=var
            )
        for var in Config.variables_nwp4:
            data[f"SELECTED_NWP4_{var}"] = fill_run_nans_w_previous_run(
                data=data, run=f"NWP4_12h_D_{var}", model="NWP4", var=var
            )
        return data

    def fit(self, X, y):
        return self

    def transform(self, X):
        X_ = X.copy()

        X_ = self._get_cyclical_variables(X_)
        X_ = self._fill_all_runs(X_)
        X_ = X_[
            [
                "sin_day",
                "cos_day",
                "sin_year",
                "cos_year",
                "SELECTED_NWP1_U",
                "SELECTED_NWP1_V",
                "SELECTED_NWP1_T",
                "SELECTED_NWP2_U",
                "SELECTED_NWP2_V",
                "SELECTED_NWP3_U",
                "SELECTED_NWP3_V",
                "SELECTED_NWP3_T",
            ]
        ]

        X_ = X_.interpolate(method="linear", limit_direction="both", axis=0)

        X_["sin_day_year"] = X_["sin_day"] * X_["sin_year"]
        X_["cos_day_year"] = X_["cos_day"] * X_["cos_year"]

        for model in ["1", "2", "3"]:
            X_[f"WIND_SPEED_NWP{model}"] = wind_components_to_speed(
                X_[f"SELECTED_NWP{model}_U"], X_[f"SELECTED_NWP{model}_V"]
            )
            X_[f"WIND_DIR_NWP{model}"] = wind_components_to_dir(
                X_[f"SELECTED_NWP{model}_U"], X_[f"SELECTED_NWP{model}_V"]
            )
            X_[f"WIND_SPEED_NWP{model}_cube"] = X_[f"WIND_SPEED_NWP{model}"] ** 3

        X_["AVG_WIND_SPEED"] = X_.loc[
            :, ["WIND_SPEED_NWP1", "WIND_SPEED_NWP2", "WIND_SPEED_NWP3"]
        ].mean(axis=1)
        X_["STD_WIND_SPEED"] = X_.loc[
            :, ["WIND_SPEED_NWP1", "WIND_SPEED_NWP2", "WIND_SPEED_NWP3"]
        ].std(axis=1)

        X_["AVG_WIND_DIR"] = X_.loc[
            :, ["WIND_DIR_NWP1", "WIND_DIR_NWP2", "WIND_DIR_NWP3"]
        ].mean(axis=1)
        X_["STD_WIND_DIR"] = X_.loc[
            :, ["WIND_DIR_NWP1", "WIND_DIR_NWP2", "WIND_DIR_NWP3"]
        ].std(axis=1)

        X_["AVG_WIND_SPEED_cube"] = X_.loc[
            :, ["WIND_SPEED_NWP1_cube", "WIND_SPEED_NWP2_cube", "WIND_SPEED_NWP3_cube"]
        ].mean(axis=1)
        X_["STD_WIND_SPEED_cube"] = X_.loc[
            :, ["WIND_SPEED_NWP1_cube", "WIND_SPEED_NWP2_cube", "WIND_SPEED_NWP3_cube"]
        ].std(axis=1)

        X_ = self._get_lagged_feature(
            X=X_, feature="AVG_WIND_SPEED", n_lags=self.n_lags
        )

        X_ = X_.interpolate(method="linear", limit_direction="both", axis=0)
        X_ = X_.drop(
            [
                "SELECTED_NWP1_U",
                "SELECTED_NWP1_V",
                "SELECTED_NWP2_U",
                "SELECTED_NWP2_V",
                "SELECTED_NWP3_U",
                "SELECTED_NWP3_V",
            ],
            axis=1,
        )

        return X_


class WindFarmPrediction:
    def __init__(
        self, base_features=None, target=None, pipeline=None,
    ):

        self.target = target
        self.pipeline = pipeline

    def fit(self, X: pd.DataFrame, y):
        return self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame):
        prediction = self.pipeline.predict(X)
        return pd.DataFrame({"prediction": prediction}, index=X.index)

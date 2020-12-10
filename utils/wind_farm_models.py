import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from utils.wind_configs import Config, MetaLogger
from utils.wind_utils import FeatureEngineering, WindFarmPrediction


class WindFarmModel:
    def __init__(self, train_data, target, feat_eng_parameters, algo_hyperparams):
        self.train_data = train_data
        self.feat_eng_parameters = feat_eng_parameters
        self.algo_hyperparams = algo_hyperparams
        self.logger = MetaLogger.setup_logging(__name__)

    def run_model(self) -> None:
        MetaLogger.log_message(self.logger, "Load wind farm data")
        self.load_data()
        MetaLogger.log_message(self.logger, "Wind farm data loaded")
        self.set_prediction_pipeline()
        MetaLogger.log_message(self.logger, f"Training model")
        self.prediction_pipeline.fit(self.X, self.y)
        MetaLogger.log_message(self.logger, f"Model trained")

    def load_data(self) -> None:
        self.X = self.train_data.drop(["Production"], axis=1)
        self.y = self.train_data.loc[:, "Production"]

    def remove_outliers(self) -> None:
        pass

    def create_prediction_pipeline(self) -> Pipeline:
        feature_engineering = FeatureEngineering(**self.feat_eng_parameters)
        x_boost = XGBRegressor(**self.algo_hyperparams["x_boost"])
        rf = RandomForestRegressor(**self.algo_hyperparams["rf"])
        vr = VotingRegressor([("x_boost", x_boost), ("rf", rf)])
        return Pipeline(
            steps=[
                ("feature_engineering", feature_engineering),
                ("voting_regressor", vr),
            ]
        )

    def set_prediction_pipeline(self) -> None:
        self.prediction_pipeline = self._init_prediction_pipeline()
        self.prediction_pipeline.pipeline = self.create_prediction_pipeline()

    def _init_prediction_pipeline(self) -> WindFarmPrediction:
        prediction_pipeline = WindFarmPrediction(target="Production", pipeline=None)
        return prediction_pipeline

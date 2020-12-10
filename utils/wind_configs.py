import json
import logging


class MetaLogger:

    LOGGER_FORMAT = "time='%(asctime)s' level=%(levelname)s event=%(message)s"

    @staticmethod
    def setup_logging(name, default_level=logging.INFO):
        """Setup logging configuration."""
        root = logging.getLogger()
        if root.handlers:
            for handler in root.handlers:
                root.removeHandler(handler)
        logging.basicConfig(format=MetaLogger.LOGGER_FORMAT)
        logger = logging.getLogger(name)
        logger.setLevel(default_level)
        return logger

    @staticmethod
    def log_message(logger, message):
        logger.info(json.dumps({"message": message}))


class Config:

    data_folder_path = "/Users/emilenaffah/Documents/data-projects/Challenges/wind-power-forecasting/data/"
    x_train_file = "X_train.csv"
    y_train_file = "y_train.csv"
    x_test_file = "X_test.csv"
    complementary_data_file = "WindFarms_complementary_data.csv"
    n_nwp = 4
    nwp13_runs = ["00", "06", "12", "18"]
    nwp24_runs = ["00", "12"]
    days_to_D = ["-2", "-1", ""]
    variables_npw13 = ["U", "V", "T"]
    variables_npw2 = ["U", "V"]
    variables_nwp4 = ["U", "V", "CLCT"]

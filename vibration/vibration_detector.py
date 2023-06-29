import json
from pathlib import Path
import logging
from joblib import dump, load
import pandas as pd

from databot.PyDatabot import PyDatabot, DatabotConfig


class DatabotVibrationDetector(PyDatabot):

    def __init__(self, databot_config: DatabotConfig, model_file_name:str, log_level: int = logging.INFO):
        super().__init__(databot_config, log_level)
        self.model = load(model_file_name)

    def process_databot_data(self, epoch, data):
        df = pd.DataFrame([data])
        df.drop(columns=['time'], inplace=True)
        prediction = self.model.predict(df)
        print(f"{data['time']}: {prediction}")


def main():
    with open("../databot_address.txt", "r") as f:
        databot_address = f.read()

    c = DatabotConfig()
    c.Laccl = True
    c.refresh = 100
    c.address = databot_address
    db = DatabotVibrationDetector(c, model_file_name="./vibration_detector_model.sav")
    db.run()


if __name__ == '__main__':
    main()

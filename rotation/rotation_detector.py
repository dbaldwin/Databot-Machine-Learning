import logging

import pandas as pd
from databot.PyDatabot import PyDatabot, DatabotConfig
from joblib import load


class PyDatabotRotationDetector(PyDatabot):
    def __init__(self, databot_config: DatabotConfig, model_file_name: str, log_level: int = logging.INFO):
        super().__init__(databot_config, log_level)
        self.model = load(model_file_name)
        self.labels = ['steady', 'pendulum', 'vertical', 'horizontal']  # 0-steady, 1-pendulum, 2-vertical, 3-horizontal

    def process_databot_data(self, epoch, data):
        df = pd.DataFrame([data])
        df.drop(columns=['time'], inplace=True)
        prediction = self.model.predict(df)
        print(f"{data['time']}: {prediction} ({self.labels[prediction[0]]})")


def main():
    c = DatabotConfig()
    c.gyro = False
    c.Laccl = True
    c.refresh = 200
    c.address = PyDatabot.get_databot_address()

    db = PyDatabotRotationDetector(databot_config=c, model_file_name="rotation_detector_model_no_horizontal.sav")
    db.run()


if __name__ == '__main__':
    main()

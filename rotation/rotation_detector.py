import sys
sys.path.append("../..")

from databot.PyDatabot import PyDatabot, DatabotConfig, DatabotBLEConfig
from joblib import dump, load
import pandas as pd

class PyDatabotRotationDetector(PyDatabot):

    async def run_queue_consumer(self):
        self.logger.info("Starting rotation detector")
        rotational_model = load('./rotation_detector_model.sav')
        labels = ['steady', 'pendulum', 'vertical', 'horizontal'] # 0-steady, 1-pendulum, 2-vertical, 3-horizontal
        while True:
            # Use await asyncio.wait_for(queue.get(), timeout=1.0) if you want a timeout for getting data.
            epoch, data = await self.queue.get()
            if data is None:
                self.logger.info(
                    "Got message from client about disconnection. Exiting consumer loop..."
                )
                break
            else:
                print(data)
                data_record = pd.DataFrame([data])
                data_record = data_record.drop(columns=['time'])
                prediction_proba = rotational_model.predict_proba(data_record)
                prediction = rotational_model.predict(data_record)
                self.logger.info(f"Rotation Prediction: {prediction[0]} ({labels[prediction[0]]}), {prediction_proba}")


def main():
    c = DatabotConfig()
    c.gyro = True
    c.Laccl = True
    c.address = PyDatabot.get_databot_address()

    db = PyDatabotRotationDetector(c)
    db.run()


if __name__ == '__main__':
    main()

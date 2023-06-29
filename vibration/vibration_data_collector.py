import json
from pathlib import Path
import logging

from databot.PyDatabot import PyDatabot, DatabotConfig


class SaveToFileDatabotCollector(PyDatabot):

    def __init__(self, databot_config: DatabotConfig, file_name: str, target_value:int, log_level: int = logging.INFO):
        super().__init__(databot_config, log_level)
        self.file_name = f"data/{file_name}"
        self.file_path = Path(self.file_name)
        if self.file_path.exists():
            self.file_path.unlink(missing_ok=True)
        self.record_number = 0
        self.target_value = target_value

    def process_databot_data(self, epoch, data):
        with self.file_path.open("a", encoding="utf-8") as f:
            data['vibration'] = self.target_value

            f.write(json.dumps(data))
            f.write("\n")
            self.logger.info(f"wrote record[{self.record_number}]")
            self.record_number = self.record_number + 1
            if self.record_number >= 1000:
                raise Exception("Done collecting data")


def main():
    with open("../databot_address.txt", "r") as f:
        databot_address = f.read()

    c = DatabotConfig()
    c.Laccl = True
    c.refresh = 100
    c.address = databot_address
    db = SaveToFileDatabotCollector(c, file_name="vibration.json", target_value=1)
    db.run()


if __name__ == '__main__':
    main()

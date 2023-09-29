from databot.PyDatabot import PyDatabot, PyDatabotSaveToFileDataCollector, DatabotConfig


def main(filename, target_value):
    c = DatabotConfig()
    c.gyro = False
    c.Laccl = True
    c.refresh = 200
    c.address = PyDatabot.get_databot_address()
    db = PyDatabotSaveToFileDataCollector(c,
                                          file_name=f"data/{filename}",
                                          number_of_records_to_collect=1000,
                                          # 0-steady, 1-pendulum, 2-vertical, 3-horizontal
                                          extra_data={
                                              "rotation": target_value
                                          }
                                          )
    db.run()


if __name__ == '__main__':
    input("Press return to start data collection...")
    # main("steady.txt", 0)
    # main("pendulum.txt", 1)
    # main("vertical.txt", 2)
    main("horizontal.txt", 3)

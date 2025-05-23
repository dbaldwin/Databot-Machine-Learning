from databot.PyDatabot import PyDatabot, DatabotConfig


def main():
    """
    An example of how to collect CO2 values from the databot using the PyDatabot API.

    :return: None
    """
    c = DatabotConfig()
    c.co2 = True
    c.address = PyDatabot.get_databot_address()
    db = PyDatabot(c)
    db.run()


if __name__ == '__main__':
    main()
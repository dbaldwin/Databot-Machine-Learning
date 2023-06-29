import asyncio

from bleak import BleakScanner


async def main():
    devices = await BleakScanner.discover()
    for d in devices:
        if d.name == "DB_databot":
            print("""Update databot/PyDatabot.py DatabotConfig class, the address attribute with the value below
            """)
            print(d.address)
            with open("./databot_address.txt", "w") as f:
                f.write(d.address)
            break
    else:
        print("The DB_databot device could not be found.  Be sure it is powered on")


if __name__ == '__main__':
    asyncio.run(main())

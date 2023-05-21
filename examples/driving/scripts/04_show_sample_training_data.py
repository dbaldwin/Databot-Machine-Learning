import pandas as pd
import argparse

direction = ['Left', 'Straight', 'Right']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=False, default=25, type=int, help="Row number of training data to display")

    args = vars(ap.parse_args())
    row = args['row']

    df = pd.read_csv('./training.csv', header=None)

    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    road_matrix = df.values
    road_data = road_matrix[row][1:]
    caption = "Label = {}, {}".format(road_matrix[row][0], direction[road_matrix[row][0]])
    imgr = road_data.reshape((10, 25))
    print(caption)
    print(imgr)

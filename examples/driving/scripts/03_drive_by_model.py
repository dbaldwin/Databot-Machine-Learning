#
# I no longer recall where I originally saw this example, but in compliance with the license you should know that
# I have changed the original source material to meet my needs
#

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except
# in compliance with the License. A copy of the License is located at
#
# https://aws.amazon.com/apache-2-0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
import sys
import csv
import random
#from sklearn.externals import joblib
import numpy as np
import random, string
import joblib

from curses import wrapper, use_default_colors, napms, curs_set, color_pair, init_pair, COLOR_RED, COLOR_WHITE, COLOR_YELLOW, COLOR_BLACK
import argparse
from pickle import load

road_centers = [0,2,3,3,2,0,-2,-3,-3,-2]
training_csv = []
car_pos = 0

def draw_road(screen):
    road_csv = []

    # needed to clear the terminal on Cloud9
    for i in range(len(road_centers)):
        screen.addstr(i+1, 1, "." * 25)
    screen.refresh()

    for i in range(len(road_centers)):
        road_center = road_centers[i]
        road_left = road_center-3;
        road_right = road_centers[i] + 3;
        line = ''
        for j in range(car_pos-12, car_pos+13):
            paint = '░'
            if j==road_left or j==road_right: paint = '█'
            if j > road_left and j < road_right: paint = '▒';
            line += paint
        road_csv.extend(list(line.replace("░","0").replace("█","1").replace("▒","2")))
        screen.addstr(i+1, 1, line)
    # draw the car
    screen.addstr(len(road_centers), 13, "▀", color_pair(1))
    screen.refresh()
    return road_csv

data_shape = ""

def predict(data):
    global data_shape
    # print(data)
    data = np.array(data).reshape(1,-1)
    data = data.astype(np.float64)
    if use_autoencode:
        scaled_data = scaler.transform(data)
        data = encoder.predict(scaled_data)

    data_shape = f"{data.shape}"
    # print(data.shape)

    y_pred = model.predict(data)
    return y_pred[0]


def main(stdscr):
    global car_pos

    use_default_colors()
    init_pair(1, COLOR_RED, COLOR_YELLOW)

    stdscr.border()
    curs_set(0)
    stdscr.addstr(1, 1, "(r) for random road")
    stdscr.addstr(2, 1, "(s) for sine wave road")
    mode = stdscr.getch()

    stdscr.nodelay(True)
    for i in range(100):
        label = "1" # straight
        road_csv = draw_road(stdscr)
        napms(300)

        # --------------------------------------------------
        # ask Model for the car direction
        # --------------------------------------------------
        label = predict(road_csv)


        if label == 0: # left
            car_pos-=1
        if label == 2: # right
            car_pos+=1

        stdscr.addstr(len(road_centers)+1, 2, "Model says: %s" % ["left    ", "straight", "right   "][label])
        stdscr.refresh()


        road_csv.insert(0, label)
        training_csv.append(road_csv)
        pop = road_centers.pop()
        if mode == ord("s"):
            road_centers.insert(0, pop)
        else:
            first = road_centers[0]
            rnd = random.choice([0,1,2])
            if rnd==0: road_centers.insert(0, first-1)
            if rnd==1: road_centers.insert(0, first)
            if rnd==2: road_centers.insert(0, first+1)


    stdscr.nodelay(False)
    stdscr.addstr(12, 1, "Press any key to quit")
    stdscr.getkey()
    print(f"Data Shape: {data_shape}")

model = None
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--autoencode", action='store_true', help="Use autoencoder")

    args = vars(ap.parse_args())

    use_autoencode = args['autoencode']
    scaler = None
    encoder = None
    if use_autoencode:
        # load the model
        encoder = load_model("encoder_model.h5")
        print(encoder.summary())
        # load the scaler
        scaler = load(open('encoder_scaler.pkl', 'rb'))

    model = joblib.load('best_driving_model.sav')

    from curses import wrapper
    wrapper(main)

    # # wwrite the trainind/driving data from the simulation
    # r_string = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(5))
    # random_training_filename = f"training_{r_string}.csv"
    #
    # with open(random_training_filename, 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for row in training_csv:
    #         writer.writerow(row)

import glob
import math
from collections import deque

import cv2
import numpy as np
from argparse import ArgumentParser
from ultralytics import YOLO

from utils import generate_rgb_color, draw_trespass_region, TRACKING_LEN, is_trespassing


tracking = {}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n", help="name of the model")
    parser.add_argument("--folder_path", type=str, default="dataset/test_sequence_1_floor", help="path to a dataset")
    parser.add_argument("--export", type=str, default="", help="export path")

    args = parser.parse_args()
    model = YOLO(f'http://localhost:8000/{args.model}', task='detect')

    if args.export:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.export, fourcc, 15, (640, 640))

    for filename in glob.glob(f'{args.folder_path}/*.png'):

        frame = cv2.imread(filename)
        frame = cv2.resize(frame, [640, 640], interpolation=cv2.INTER_AREA)
        frame = (cv2.normalize(frame, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255).astype("uint8")

        results = model.predict(frame, verbose=False)
        frame = draw_trespass_region(frame)
        people = 0
        trespassing = 0
        for r in results:
            for xyxy in r.boxes.xyxy.numpy():
                people += 1
                x_min, y_min, x_max, y_max = xyxy
                x = (x_max - x_min) / 2 + x_min
                y = (y_max - y_min) / 2 + y_min
                appended = False
                min_distance = float("inf")
                min_key = None
                for key, track in tracking.items():
                    current_point = np.array([x, y])
                    track_data = np.array(track)[-1]
                    distance = math.hypot(x - track_data[0], y - track_data[1])
                    if distance < min_distance:
                        min_distance = distance
                        min_key = key
                        min_track = track
                if min_distance < 20:
                    tracking[min_key].append([int(x), int(y)])
                    appended = True
                    if is_trespassing(x, y):
                        trespassing += 1
                if not appended:
                    key = generate_rgb_color()
                    track = deque([[int(x), int(y)]], TRACKING_LEN)
                    tracking[key] = track
                    min_key = key
                    min_track = track
                    if is_trespassing(x, y):
                        trespassing += 1
                frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), min_key, 4)
                if len(track) > 1:
                    points = np.array(list(min_track), np.int32).reshape((-1, 1, 2))
                    frame = cv2.polylines(frame, [points], False, min_key, 5)
        txt_col = (0, 255, 0) if trespassing == 0 else (0, 0, 255)
        frame = cv2.putText(frame, f'People: {people}', (25, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, f'Trespassing: {trespassing}', (400, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, txt_col, 2, cv2.LINE_AA)

        print(f'Path: {filename}, People: {people}, Trespassing: {trespassing}')
        cv2.imshow("Frame", frame)
        if args.export:
            video_writer.write(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    if args.export:
        video_writer.release()

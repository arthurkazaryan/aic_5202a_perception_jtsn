import glob
from collections import deque

import cv2
import numpy as np
from argparse import ArgumentParser
from ultralytics import YOLO

from utils import generate_rgb_color, TRACKING_LEN

# tracking = {0: deque([], 3)}
tracking = {}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n", help="name of the model")
    parser.add_argument("--folder_path", type=str, default="dataset/test_sequence_2_normalized", help="path to a dataset")
    parser.add_argument("--export", type=str, default="", help="export path")

    args = parser.parse_args()
    model = YOLO(f'http://localhost:8000/{args.model}', task='detect')

    if args.export:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.export, fourcc, 15, (640, 640))

    for filename in glob.glob(f'{args.folder_path}/*.png'):

        frame = cv2.imread(filename)
        frame = cv2.resize(frame, [640, 640], interpolation=cv2.INTER_AREA)
        # frame = cv2.normalize(frame, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255

        results = model.predict(frame)

        if len(results) > 0:
            for r in results:
                for xyxy in r.boxes.xyxy.numpy():
                    x_min, y_min, x_max, y_max = xyxy
                    x = (x_max - x_min) / 2 + x_min
                    y = (y_max - y_min) / 2 + y_min
                    appended = False
                    min_distance = float("inf")
                    min_key = None
                    for key, track in tracking.items():
                        current_point = np.array([x, y])
                        mean_track_data = np.array(track).mean(axis=0)
                        distance = np.linalg.norm(current_point - mean_track_data)
                        if distance < min_distance:
                            min_distance = distance
                            min_key = key
                        # if abs(x - track[0][0]) < 80 and abs(y - track[0][1]) < 80:
                        #     tracking[key].append([int(x), int(y)])
                        #     appended = True
                        #     break
                    if min_distance < 50:
                        tracking[min_key].append([int(x), int(y)])
                        appended = True
                    if not appended:
                        key = generate_rgb_color()
                        track = deque([[int(x), int(y)]], TRACKING_LEN)
                        tracking[key] = track
                    # res_plotted = r.plot()  # allows you to retrieve the prediction results
                    frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), key, 4)
                    if len(track) > 1:
                        points = np.array(list(track), np.int32).reshape((-1, 1, 2))
                        frame = cv2.polylines(frame, [points], False, key, 5)
        cv2.imshow("Frame", frame)
        if args.export:
            video_writer.write(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    if args.export:
        video_writer.release()

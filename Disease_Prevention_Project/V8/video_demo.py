import cv2
import face_recognition
import datetime as dt
from collections import Counter
import itertools
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import multiprocessing as mp
import psutil
from dataclasses import dataclass
from typing import List, Optional
import statistics
import math
import functools as ft
# import subprocess
import threading
import os
# import yappi

from set_camera import set_cam
from get_db import get_user_profiles
from Combine_database import fetch_newest_temperature_db
from get_file_path import get_file_path
from Insert_Measure_Info import Insert_Measure_Info
from Update_User_Photo import Update_User_Photo
import camera


def execute_FLIR():
    # Execute parent folder's .bat file
    FLIR_start = str(Path(get_file_path()).parent.joinpath("Start_temp.bat"))
    flag = os.system(FLIR_start)
    # process = subprocess.Popen(FLIR_start, stderr=subprocess.PIPE)
    print("end of subprocess call")
    if not flag:  # 把 exe 執行出來的結果讀回來
        print("**************************************************************")
        print('error execute_FLIR')
        print("**************************************************************")


@dataclass(frozen=True)
class RecognitionResult():
    name: str
    person_id: str
    distance: float
    matched: bool


# Return the index of the minimum distance
def find_closest(distances):
    return distances.argmin()


# Get the indices of faces in a list of frames which
#  corresponding to each face in the last frame.
def same_face_indices(prev_encodings, prev_locations, tolerance):
    selected_bools = []
    for col, x in enumerate(prev_encodings):
        selected_bools.append([])
        for y in x:
            selected_bools[col].append(False)
    same_indices = [[] for x in range(len(prev_encodings[-1]))]
    for i, (last_encoding, last_location, indices) \
            in enumerate(zip(
                prev_encodings[-1],
                prev_locations[-1],
                same_indices
            )):
        for row, (encodings, locations, bools)\
                in enumerate(zip(
                    prev_encodings[:-1],
                    prev_locations[:-1],
                    selected_bools[:-1]
                )):
            closest = None
            min_distance = math.inf
            for col in range(len(encodings)):
                distance = face_recognition.face_distance(
                    [encodings[col]], last_encoding
                )
                if distance < min_distance and distance <= tolerance\
                        and bools[col] is False:
                    closest = col
                    min_distance = distance
            if closest is not None:
                bools[closest] = True
                indices.append((row, closest))
        indices.append((len(prev_encodings) - 1, i))
    return same_indices


def info_hud(frame_size, font, detect_rect_):
    detect_top, detect_right, detect_bottom, detect_left = detect_rect_
    hud = Image.new('RGBA', frame_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(hud)
    textColor = (0, 0, 255)
    stroke_color = (255, 255, 255)
    draw.text((detect_left + 10, detect_bottom + 10),
              '按s更新圖片',
              font=font, fill=textColor,
              stroke_fill=stroke_color, stroke_width=2)
    return hud


def result_hud(frame_size, locations, results: List[RecognitionResult],
               tolerance, font, detect_rect_):
    hud = Image.new('RGBA', frame_size, (255, 255, 255, 0))
    rectColor = (0, 0, 255, 128)
    draw = ImageDraw.Draw(hud)
    top, right, bottom, left = detect_rect_
    xs = [left, right, left, right]
    ys = [top, top, bottom, bottom]
    x_factors = [1 if x == left else -1 for x in xs]
    y_factors = [1 if y == top else -1 for y in ys]
    for x_factor, y_factor, x, y in zip(x_factors, y_factors, xs, ys):
        draw.line([(x, y + 20 * y_factor), (x, y), (x + 20 * x_factor, y)],
                  fill=rectColor, width=3)
    for (top, right, bottom, left), result in zip(locations, results):
        rectColor = (0, 255, 0, 192) if result.matched else (255, 0, 0, 192)
        draw.rectangle([(left, top), (right, bottom)],
                       fill=None, outline=rectColor, width=5)
        draw.rectangle([(left, bottom), (right, bottom + 50)],
                       fill=rectColor, outline=None)
        textColor = (0, 0, 0)
        draw.text((left + 6, bottom), f'{result.name}',
                  font=font, fill=textColor)
        draw.text((left + 6, bottom + 20), f'差異 {result.distance: > .3f}',
                  font=font, fill=textColor)
    return hud


# Return (top, left, bottom, right)
def detect_rect(frame_size, ratio):
    frame_width, frame_height = frame_size
    return (int(frame_height / ratio),
            int(frame_width * (ratio - 1) / ratio),
            int(frame_height * (ratio - 1) / ratio),
            int(frame_width / ratio))


def in_detect_range(face_rect, detect_rect_):
    detect_top, detect_right, detect_bottom, detect_left = detect_rect_
    top, right, bottom, left = face_rect
    return (top > detect_top
            and right < detect_right
            and bottom < detect_bottom
            and left > detect_left)


def is_new_person(time_dict, person_id, now):
    span = 5.0 if person_id == 'guest' else 10.0
    return (now - time_dict.setdefault(
        person_id, dt.datetime.min)).total_seconds() > span


def encode(frame, locations):
    encodings = face_recognition.face_encodings(
        frame, locations)
    return encodings


# Push new elements into the given list,
#  and pop old elements like a queue.
def push_list_queue(list_: list, new_elements: list, queue_size: int):
    size_sum = len(list_) + len(new_elements)
    start = 0 if size_sum < queue_size else size_sum - queue_size
    return list_[start:] + new_elements


def get_available_cpu_count():
    return len(psutil.Process().cpu_affinity())


# Get the indices of encodings that have the most matching
def detection_results(user_profiles, prev_locations, prev_encodings,
                      prev_distances, prev_closest_ids, tolerance,
                      min_matched_ratio) -> Optional[List[RecognitionResult]]:
    indices_list = same_face_indices(prev_encodings, prev_locations, tolerance)
    if False in [len(x) >= len(prev_encodings) / 2 for x in indices_list]:
        return None
    results = []
    for indices in indices_list:
        distances = [prev_distances[row][col] for row, col in indices]
        closest_ids = [prev_closest_ids[row][col] for row, col in indices]
        closest_distances = [distance[closest_id] for distance, closest_id
                             in zip(distances, closest_ids)]
        matched_indices = [i for i, closest_distance
                           in enumerate(closest_distances)
                           if closest_distance <= tolerance]
        matched_ratio = len(matched_indices) / len(indices)
        matched = matched_ratio >= min_matched_ratio
        if matched:
            matched_closest_ids = [closest_ids[i] for i in matched_indices]
            matched_closest_distances = [closest_distances[i]
                                         for i in matched_indices]
            id_counts = Counter(matched_closest_ids).most_common()
            most_ids = [e for e, count in id_counts
                        if count == id_counts[0][1]]
            min_mean_distance = math.inf
            min_mean_distance_id = None
            for most_id in most_ids:
                mean = statistics.mean([matched_closest_distances[i] for i, id_
                                        in enumerate(matched_closest_ids)
                                        if id_ == most_id])
                if mean < min_mean_distance:
                    min_mean_distance = mean
                    min_mean_distance_id = most_id
            profile = user_profiles[min_mean_distance_id]
            person_id = profile[1]
            name = profile[2]
        else:
            person_id = 'guest'
            name = '訪客'
        distance = sorted(closest_distances)[
            int(math.ceil(len(indices) * min_matched_ratio)) - 1]
        results.append(RecognitionResult(name, person_id, distance, matched))
    return results


# Insert measure into database
def insert_measure(result: RecognitionResult, db_name):
    data = fetch_newest_temperature_db(db_name)
    measure_info_profile = [result.person_id] + list(data)
    Insert_Measure_Info(db_name, measure_info_profile)


# 若根據偵測時間判斷為新的人，將資料寫進資料庫
def record_new_measures(time_dict, now, results: List[RecognitionResult],
                        db_name):
    new_results = [x for x in results
                   if is_new_person(time_dict, x.person_id, now)]
    for result in new_results:
        insert_measure(result, db_name)


def frame_buffer_size(prev_times, now, buffer_duration, cpu_count):
    buffer_size = cpu_count
    for i, time in enumerate(prev_times):
        if (now - time).total_seconds() < buffer_duration:
            buffer_size = max(buffer_size, len(prev_times) - i + 1)
            break
    return buffer_size


def main():
    # Fix pyinstaller's bug
    mp.freeze_support()
    mp.set_start_method('spawn')

    video_num = set_cam()
    if video_num is None:
        print("未選擇攝影機")
        return
    t = threading.Thread(target=execute_FLIR)
    t.daemon = True
    t.start()
    # execute_FLIR()

    database_name = './Release/teacher.db'
    tolerance = 0.38
    min_matched_ratio = 0.3
    buffer_duration = 2  # Frames within this duration will be buffered
    # [[en_face, ID, Name], ...]
    user_profiles = get_user_profiles(database_name)
    known_face_encodings = [x[0] for x in user_profiles]

    prev_locations = []  # Previous face locations in buffered frames
    prev_frames = []
    prev_times = []
    prev_encodings = []  # Previous face encodings in buffered frames
    prev_distances = []
    prev_closest_ids = []
    time_dict = {}  # {person_id: leaving_time}
    last_record_time = dt.datetime.min
    results: List[RecognitionResult] = []
    locations = []

    cpu_count = get_available_cpu_count()
    pool = mp.Pool(processes=cpu_count)

    frame_scale = 0.5
    main_camera = camera.Camera(video_num)
    frame = main_camera.read()
    video_height = int(frame.shape[0])
    video_width = int(frame.shape[1])
    detect_edge_ratio = 12
    detect_rect_ = detect_rect((video_width, video_height),
                               detect_edge_ratio)
    font = ImageFont.truetype(
        str(Path(get_file_path()).joinpath('NotoSansCJK-Regular222.ttc')), 18)

    for frame_count in itertools.count():
        while True:
            input_key = cv2.waitKey(1)
            # Press q to quit
            if input_key & 0xFF == ord('q'):
                pool.terminate()
                pool.join()
                main_camera.stop()
                print('video demo ended')
                return

            frame = main_camera.read()
            # press s to update database and save pictures
            if input_key & 0xFF == ord('s'):
                Update_User_Photo(database_name, frame)
                user_profiles = get_user_profiles(database_name)
                known_face_encodings = [x[0] for x in user_profiles]
            info_hud_ = info_hud(
                (video_width, video_height), font, detect_rect_)
            result_hud_ = result_hud((video_width, video_height),
                                     [tuple(y / frame_scale for y in x)
                                      for x in locations],
                                     results, tolerance, font, detect_rect_)
            rgba_frame = Image.fromarray(frame).convert('RGBA')
            result_frame = np.asarray(Image.alpha_composite(
                Image.alpha_composite(rgba_frame, info_hud_),
                result_hud_
            ))
            cv2_result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', cv2_result_frame)
            small_frame = cv2.resize(
                frame, (0, 0), fx=frame_scale, fy=frame_scale)
            new_locations = [
                x for x in face_recognition.face_locations(small_frame)
                if in_detect_range([y / frame_scale for y in x], detect_rect_)]
            if len(new_locations) > 0 or (dt.datetime.now() - last_record_time
                                          ).total_seconds() > buffer_duration:
                break
        last_record_time = dt.datetime.now()
        locations = new_locations
        now = dt.datetime.now()
        buffer_size = frame_buffer_size(
            prev_times, now, buffer_duration, cpu_count)
        push_buffer = ft.partial(push_list_queue, queue_size=buffer_size)
        prev_times = push_buffer(prev_times, [now])
        prev_locations = push_buffer(prev_locations, [locations])
        prev_frames = push_buffer(prev_frames, [small_frame])
        # Generate results every cpu_count frames
        if frame_count % cpu_count == cpu_count - 1:
            # last = dt.datetime.now()
            encodings = pool.starmap(
                encode,
                [(prev_frames[i], prev_locations[i])
                 for i in range(-cpu_count, 0)]
            )
            # print(dt.datetime.now() - last)
            distances = [[face_recognition.face_distance(
                known_face_encodings, x) for x in y] for y in encodings]
            matched_ids = [[find_closest(x) for x in y] for y in distances]
            prev_encodings = push_buffer(prev_encodings, encodings)
            prev_distances = push_buffer(prev_distances, distances)
            prev_closest_ids = push_buffer(prev_closest_ids, matched_ids)
            new_results = detection_results(
                user_profiles, prev_locations, prev_encodings,
                prev_distances, prev_closest_ids, tolerance,
                min_matched_ratio
            )
            if new_results is not None:
                results = new_results
            now = dt.datetime.now()
            record_new_measures(time_dict, now, results, database_name)
            for result in results:
                time_dict[result.person_id] = now


if __name__ == '__main__':
    main()

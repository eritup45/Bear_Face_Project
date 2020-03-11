import cv2
from datetime import datetime
import face_recognition
import datetime as dt
from collections import Counter
import itertools
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import sys
import multiprocessing as mp
import psutil
# import yappi

from get_db import get_user_profiles
from Combine_database import fetch_newest_temperature_db
from get_file_path import get_file_path
from Insert_Measure_Info import Insert_Measure_Info
from Update_User_Photo import Update_User_Photo

# Return:
# id: the id of the minimum distance


def find_closest(list_of_face_encodings, unknown_face_encoding):
    distances = face_recognition.face_distance(
        list_of_face_encodings, unknown_face_encoding)
    id = distances.argmin()
    return id


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
            min_distance = 1.0
            for col in range(len(encodings)):
                distance = face_recognition.face_distance(
                    [encodings[col]], last_encoding
                )
                if distance < min_distance and distance < tolerance\
                        and bools[col] is False:
                    closest = col
                    min_distance = distance
            if closest is not None:
                bools[closest] = True
                indices.append((row, closest))
        indices.append((len(prev_encodings) - 1, i))
    return same_indices


def draw_results(frame, locations, results, tolerance,
                 font, scale, detect_rect_):
    rgb_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detect_top, detect_right, detect_bottom, detect_left = [
        x * 2 for x in detect_rect_]
    rectColor = (0, 0, 255)
    draw = ImageDraw.Draw(rgb_frame)
    draw.rectangle([(detect_left, detect_top), (detect_right, detect_bottom)],
                   fill=None, outline=rectColor, width=5)
    for (top, right, bottom, left), (name, _, distance)\
            in zip(locations, results):
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        match_state = distance < tolerance
        rectColor = (0, 255, 0) if match_state else (255, 0, 0)
        draw.rectangle([(left, top), (right, bottom)],
                       fill=None, outline=rectColor, width=5)

        draw.rectangle([(left, bottom), (right, bottom + 50)],
                       fill=rectColor, outline=None)
        textColor = (0, 0, 0)
        draw.text((left + 6, bottom),
                  str(f'{name}, Diff: {distance: > .4f}'),
                  font=font, fill=textColor)
    result_frame = cv2.cvtColor(np.asarray(rgb_frame), cv2.COLOR_RGB2BGR)
    return result_frame


def detect_rect(frame_size, ratio):
    frame_height, frame_width = frame_size
    return (frame_height / ratio,
            frame_width * (ratio - 1) / ratio,
            frame_height * (ratio - 1) / ratio,
            frame_width / ratio)


def in_detect_range(face_rect, detect_rect_):
    detect_top, detect_right, detect_bottom, detect_left = detect_rect_
    top, right, bottom, left = face_rect
    return (top > detect_top
            and right < detect_right
            and bottom < detect_bottom
            and left > detect_left)


def is_new_person(time_dict, name, now):
    span = 5.0 if name == '訪客' else 10.0
    return (now - time_dict.setdefault(name, dt.datetime.min))\
        .total_seconds() > span


def encode(frame, locations):
    encodings = face_recognition.face_encodings(frame, locations)
    return encodings


# def encode(encode_request_queue: mp.Queue, encode_result_queue: mp.Queue):
#     while True:
#         (frame, locations, order) = encode_request_queue.get()
#         encodings = face_recognition.face_encodings(frame, locations)
#         encode_result_queue.put((encodings, order))

# Push new elements into the given list,
#  and pop old elements like a queue.
def push_list_queue(list_: list, new_elements: list, queue_size: int):
    size_sum = len(list_) + len(new_elements)
    start = (0 if size_sum < queue_size
             else size_sum - queue_size)
    list_ = list_[start:] + new_elements
    return list_


def main():
    # Fix pyinstaller's bug
    mp.freeze_support()
    mp.set_start_method('spawn')

    if len(sys.argv) >= 3:
        video_num = int(sys.argv[2])
    else:
        video_num = 0
    database_name = './Release/teacher.db'
    tolerance = 0.385
    buffer_duration = 2  # Frames within this duration will be buffered
    # [[en_face, ID, Name], ...]
    user_profiles = get_user_profiles(database_name)
    known_face_encodings = [x[0] for x in user_profiles]

    prev_locations = []  # Previous face locations in buffered frames
    prev_encodings = []  # Previous face encodings in buffered frames
    # Indices of previous matched encodings in buffered frames
    prev_matched_ids = []
    prev_frames = []
    prev_times = []
    time_dict = {}  # {id: leaving_time}
    last_record_time = dt.datetime.min
    results = []  # [[closest_id, distance], ...]
    locations = []
    record_frame_count = 0

    available_cpus = psutil.Process().cpu_affinity()
    cpu_count = len(available_cpus)
    encode_request_queue = mp.Queue(maxsize=cpu_count)
    encode_result_queue = mp.Queue()
    pool = mp.Pool(processes=cpu_count)

    video_capture = cv2.VideoCapture(video_num)
    frame_scale = 0.5
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * frame_scale
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) * frame_scale
    detect_edge_ratio = 6.5
    detect_rect_ = detect_rect((video_height, video_width), detect_edge_ratio)
    font = ImageFont.truetype(
        str(Path(get_file_path()).joinpath('NotoSansCJK-Regular222.ttc')), 20)

    for frame_count in itertools.count():
        while True:
            input_key = cv2.waitKey(1)
            # Press q to quit
            if input_key & 0xFF == ord('q'):
                return
            _, frame = video_capture.read()
            # press s to update database and save pictures
            if input_key & 0xFF == ord('s'):
                Update_User_Photo(database_name, frame)
                user_profiles = get_user_profiles(database_name)
                known_face_encodings = [x[0] for x in user_profiles]
            small_frame = cv2.resize(
                frame, (0, 0), fx=frame_scale, fy=frame_scale)
            rgb_frame = small_frame[:, :, ::-1]
            result_frame = draw_results(
                frame, locations, results, tolerance,
                font, 1 / frame_scale, detect_rect_)
            cv2.imshow('Video', result_frame)
            new_locations = [x for x
                             in face_recognition.face_locations(rgb_frame)
                             if in_detect_range(x, detect_rect_)]
            if len(locations) > 0 or (dt.datetime.now() - last_record_time
                                      ).total_seconds() > 2.0:
                break
        record_frame_count += 1
        last_record_time = dt.datetime.now()
        locations = new_locations
        now = dt.datetime.now()
        frame_buffer_size = 1
        for i, time in enumerate(prev_times):
            if (now - time).total_seconds() < buffer_duration:
                frame_buffer_size = len(prev_times) - i + 1
                break
        frame_buffer_size = max(frame_buffer_size, cpu_count)
        prev_times = push_list_queue(prev_times, [now], frame_buffer_size)
        prev_locations = push_list_queue(
            prev_locations, [locations], frame_buffer_size)
        prev_frames = push_list_queue(
            prev_frames, [rgb_frame], frame_buffer_size)
        # Generate results every cpu_count frames
        if record_frame_count % cpu_count == 0:
            print(frame_buffer_size)
            encodings = pool.starmap(
                encode,
                [(prev_frames[i], prev_locations[i])
                 for i in range(-cpu_count, 0)]
            )
            matched_ids = [[find_closest(known_face_encodings, x)
                            for x in y] for y in encodings]
            prev_encodings = push_list_queue(
                prev_encodings, encodings, frame_buffer_size)
            prev_matched_ids = push_list_queue(
                prev_matched_ids, matched_ids, frame_buffer_size)
            # Get the indices of encodings that have the most matching
            indices_list = same_face_indices(
                prev_encodings, prev_locations, tolerance)
            if False not in [len(x) >= len(prev_encodings) // 2
                             for x in indices_list]:
                results = []
                for encoding, location, indices in zip(
                        encodings[-1], locations, indices_list):
                    ids = [prev_matched_ids[row][col] for row, col in indices]
                    id = Counter(ids).most_common(1)[0][0]
                    distance = face_recognition.face_distance(
                        (known_face_encodings)[id: id+1], encoding)[0]
                    person_id = ('guest' if distance > tolerance
                                 else user_profiles[id][1])
                    name = ('訪客' if distance > tolerance
                            else user_profiles[id][2])
                    results.append((name, person_id, distance))
            now = datetime.now()
            new_people = [(name, id) for name, id, _ in results
                          if is_new_person(time_dict, name, now)]
            old_people = [(name, id) for name, id, _ in results
                          if not is_new_person(time_dict, name, now)]
            # 若根據偵測時間判斷為新的人，將資料寫進資料庫
            for name, id in new_people:
                data = fetch_newest_temperature_db(database_name)
                measure_info_profile = [id] + list(data)
                Insert_Measure_Info(database_name, measure_info_profile)
                print(f'寫入:{name}')

            for name, _ in old_people:
                print(f'不寫入:{name}')

            for name, _, _ in results:
                time_dict[name] = now
    pool.join()
    print('Main ended')


if __name__ == '__main__':
    main()

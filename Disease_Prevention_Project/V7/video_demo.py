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
            min_distance = 1.0
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


def result_hud(frame_size, locations, results, tolerance,
               font, location_scale, detect_rect_):

    hud = Image.new('RGBA', frame_size, (255, 255, 255, 0))
    detect_top, detect_right, detect_bottom, detect_left = [
        x * location_scale for x in detect_rect_]
    rectColor = (0, 0, 0, 255)
    draw = ImageDraw.Draw(hud)
    draw.rectangle([(0, detect_bottom),
                    (frame_size[0], frame_size[1])],
                   fill=rectColor)

    for (top, right, bottom, left), (name, _, distance)\
            in zip(locations, results):
        top *= location_scale
        right *= location_scale
        bottom *= location_scale
        left *= location_scale

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
    return hud


def detect_rect(frame_size, ratio):
    frame_width, frame_height = frame_size
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
    return (now - time_dict.setdefault(
        name, dt.datetime.min)).total_seconds() > span


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
    prev_frames = []
    prev_times = []
    prev_encodings = []  # Previous face encodings in buffered frames
    # Indices of previous matched encodings in buffered frames
    prev_distances = []
    prev_matched_ids = []
    time_dict = {}  # {id: leaving_time}
    last_record_time = dt.datetime.min
    results = []  # [[closest_id, distance], ...]
    locations = []
    record_frame_count = 0

    available_cpus = psutil.Process().cpu_affinity()
    cpu_count = len(available_cpus)
    pool = mp.Pool(processes=cpu_count)

    video_capture = cv2.VideoCapture(video_num)
    frame_scale = 0.5
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    detect_edge_ratio = 12
    detect_rect_ = detect_rect(
        (int(video_width * frame_scale), int(video_height * frame_scale)),
        detect_edge_ratio)
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
            hud = result_hud((video_width, video_height), locations,
                             results, tolerance, font, 1 / frame_scale,
                             detect_rect_)
            rgb_frame = Image.fromarray(cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
            result_frame = Image.alpha_composite(rgb_frame, hud)
            cv2_result_frame = cv2.cvtColor(
                np.asarray(result_frame), cv2.COLOR_RGB2BGR)
            detect_rect_scaled = [int(x / frame_scale) for x in detect_rect_]
            cv2.imshow('Video', cv2_result_frame[
                detect_rect_scaled[0]:,
                detect_rect_scaled[3]:detect_rect_scaled[1]])
            small_frame = cv2.resize(
                frame, (0, 0), fx=frame_scale, fy=frame_scale)
            rgb_small_frame = small_frame[:, :, ::-1]
            new_locations = [x for x in face_recognition.face_locations(
                rgb_small_frame) if in_detect_range(x, detect_rect_)]
            if len(new_locations) > 0 or (dt.datetime.now() - last_record_time
                                          ).total_seconds() > buffer_duration:
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
            prev_frames, [rgb_small_frame], frame_buffer_size)
        # Generate results every cpu_count frames
        if record_frame_count % cpu_count == 0:
            encodings = pool.starmap(
                encode,
                [(prev_frames[i], prev_locations[i])
                 for i in range(-cpu_count, 0)]
            )
            distances = [[face_recognition.face_distance(
                known_face_encodings, x) for x in y] for y in encodings]
            matched_ids = [[find_closest(x) for x in y] for y in distances]
            prev_encodings = push_list_queue(
                prev_encodings, encodings, frame_buffer_size)
            prev_distances = push_list_queue(
                prev_distances, distances, frame_buffer_size)
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
                    matched_indices = [
                        (row, col) for row, col in indices
                        if prev_distances[row][col][
                            prev_matched_ids[row][col]] <= tolerance]
                    if len(matched_indices) > 0:
                        ids = [prev_matched_ids[row][col]
                               for row, col in matched_indices]
                        id = Counter(ids).most_common(1)[0][0]
                        person_id = user_profiles[id][1]
                        name = user_profiles[id][2]
                        distance = np.mean(
                            [prev_distances[row][col][
                                prev_matched_ids[row][col]]
                             for row, col in matched_indices
                             if prev_matched_ids[row][col] == id]
                        )
                    else:
                        person_id = 'guest'
                        name = '訪客'
                        distance = min([prev_distances[row][col].min()
                                        for row, col in indices])
                    results.append((name, person_id, distance))
            now = datetime.now()
            new_people = [(name, id) for name, id, _ in results
                          if is_new_person(time_dict, name, now)]
            # 若根據偵測時間判斷為新的人，將資料寫進資料庫
            for name, id in new_people:
                data = fetch_newest_temperature_db(database_name)
                measure_info_profile = [id] + list(data)
                Insert_Measure_Info(database_name, measure_info_profile)
                print(f'寫入:{name}')

            # for name, _ in old_people:
            #     print(f'不寫入:{name}')

            for name, _, _ in results:
                time_dict[name] = now
    pool.join()
    print('Main ended')


if __name__ == '__main__':
    main()

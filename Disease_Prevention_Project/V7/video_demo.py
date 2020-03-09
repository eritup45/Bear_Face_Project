import cv2
from datetime import datetime
from get_db import get_user_profiles
import face_recognition
import datetime as dt
from collections import Counter
import itertools
from Insert_Measure_Info import Insert_Measure_Info
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import sys
from Combine_database import fetch_newest_temperature_db, Update_Measure_Info
from get_file_path import get_file_path

# Return:
# min__: smallest distance
# id: the id of the min_distance


def find_closest(list_of_face_encodings, unknown_face_encoding):
    distances = face_recognition.face_distance(
        list_of_face_encodings, unknown_face_encoding)
    min_distance = min(distances)
    id = distances.argmin()
    return id, min_distance


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


def draw_results(frame, locations, results, user_names,
                 tolerance, font, scale, detect_rect_):
    rgb_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detect_top, detect_right, detect_bottom, detect_left = [
        x * 2 for x in detect_rect_]
    rectColor = (0, 0, 255)
    draw = ImageDraw.Draw(rgb_frame)
    draw.rectangle([(detect_left, detect_top), (detect_right, detect_bottom)],
                   fill=None, outline=rectColor, width=5)
    for (top, right, bottom, left), (_, distance), user_name \
            in zip(locations, results, user_names):
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        match_state = distance < tolerance
        rectColor = (0, 255, 0) if match_state else (255, 0, 0)
        draw.rectangle([(left, top), (right, bottom)],
                       fill=None, outline=rectColor, width=5)

        draw.rectangle([(left, bottom - 70), (right, bottom)],
                       fill=rectColor, outline=None)
        textColor = (0, 0, 0)
        draw.text((left + 6, bottom - 72),
                  str(f'{user_name}, Diff: {distance: > .4f}'),
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


def main():
    if len(sys.argv) >= 3:
        video_num = int(sys.argv[2])
    else:
        video_num = 0
    database_name = './Release/teacher.db'
    tolerance = 0.385
    frame_buffer_size = 6  # number of buffered frames to generate result
    # [[en_face, ID, Name], ...]
    user_profile_list = get_user_profiles(database_name)
    known_face_encodings = [x[0] for x in user_profile_list]

    font = ImageFont.truetype(
        str(Path(get_file_path()).joinpath('NotoSansCJK-Regular222.ttc')), 20)
    prev_locations = []  # Previous face locations in buffered frames
    prev_encodings = []  # Previous face encodings in buffered frames
    # Indices of previous matched encodings in buffered frames
    prev_matched_ids = []
    time_dict = {}  # {id: leaving_time}
    last_detected_time = dt.datetime.min
    results = []  # [[closest_id, distance], ...]

    user_names = []  # name of detected people in the current frame
    record_frame_count = 0

    video_capture = cv2.VideoCapture(video_num)
    frame_scale = 0.5
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * frame_scale
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) * frame_scale

    detect_edge_ratio = 6.5
    detect_rect_ = detect_rect((video_height, video_width), detect_edge_ratio)
    for frame_count in itertools.count():
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)
        rgb_frame = small_frame[:, :, ::-1]
        new_locations = [x for x in face_recognition.face_locations(rgb_frame)
                         if in_detect_range(x, detect_rect_)]
        if len(new_locations) > 0 or (dt.datetime.now() - last_detected_time
                                      ).total_seconds() > 2.0:
            locations = new_locations
            last_detected_time = dt.datetime.now()
            encodings = face_recognition.face_encodings(rgb_frame, locations)
            matched_ids = []
            for encoding in encodings:
                id, distance = find_closest(known_face_encodings, encoding)
                matched_ids.append(id)
            start = 0 if len(prev_locations) < frame_buffer_size else 1
            prev_locations[0:] = prev_locations[start:] + [locations]
            prev_encodings[0:] = prev_encodings[start:] + [encodings]
            prev_matched_ids[0:] = prev_matched_ids[start:] + [matched_ids]
            record_frame_count += 1
            # Generate results every 2 frames
            if record_frame_count % 2 == 0:
                # Get the indices of encodings that have the most matching
                indices_list = same_face_indices(
                    prev_encodings, prev_locations, tolerance)
                if False not in [len(x) >= 4 for x in indices_list]:
                    results = []
                    for encoding, location, indices in zip(
                            encodings, locations, indices_list):
                        ids = [prev_matched_ids[row][col]
                               for row, col in indices]
                        id = Counter(ids).most_common(1)[0][0]
                        distance = face_recognition.face_distance(
                            (known_face_encodings)[id: id+1], encoding)[0]
                        results.append((id, distance))
                user_names = ['訪客' if distance > tolerance
                              else user_profile_list[id][2]
                              for id, distance in results]
                for user_name in user_names:
                    # 若現在距離人臉最後偵測時間大於十秒，將資料寫進資料庫
                    now = datetime.now()
                    time_span = 5.0 if user_name == '訪客' else 10.0
                    if (now - time_dict.setdefault(
                        user_name, dt.datetime.min))\
                            .total_seconds() > time_span:
                        Insert_Measure_Info(database_name, [user_name, now])
                        data = fetch_newest_temperature_db(database_name)
                        Update_Measure_Info(database_name, data)
                        print(f'寫入:{user_name}')
                    else:
                        print(f'不寫入:{user_name}')
                    time_dict[user_name] = now
        # TODO: Match locations with results
        result_frame = draw_results(
            frame, locations, results, user_names,
            tolerance, font, 1 / frame_scale, detect_rect_)
        cv2.imshow('Video', result_frame)
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()

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
from Combine_database import fetch_newest_temperature_db, Update_Measure_Info

# Return:
# min_distance: smallest distance
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


def draw_results(frame, locations, results, user_names, tolerance, font, scale=2):
    rgb_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for (top, right, bottom, left), (_, min_distance), user_name \
            in zip(locations, results, user_names):
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        draw = ImageDraw.Draw(rgb_frame)
        rectColor = (255, 0, 0)
        draw.rectangle([(left, top), (right, bottom)],
                       fill=None, outline=rectColor, width=5)

        draw.rectangle([(left, bottom - 70), (right, bottom)],
                       fill=rectColor, outline=None)
        textColor = (255, 255, 255)
        draw.text((left + 6, bottom - 72),
                  str(f'{user_name}, Diff: {min_distance: > .4f}'),
                  font=font, fill=textColor)
        match_state = "Not Matched" if min_distance > tolerance else "Matched"
        draw.text((left + 6, bottom - 36),
                  str(f' {match_state}'),
                  font=font, fill=textColor)
    result_frame = cv2.cvtColor(np.asarray(rgb_frame), cv2.COLOR_RGB2BGR)
    return result_frame


def main():
    database_name = './teacher.db'
    tolerance = 0.35
    frame_buffer_size = 10  # number of buffered frames to generate result
    # [[en_face, ID, Name], ...]
    user_profile_list = get_user_profiles(database_name)
    known_face_encodings = [x[0] for x in user_profile_list]
    guest_encodings = []
    font = ImageFont.truetype(
        str(Path(__file__).parent.joinpath('NotoSansCJK-Regular222.ttc')),
        20)
    prev_locations = []  # Previous face locations in buffered frames
    prev_encodings = []  # Previous face encodings in buffered frames
    # Indices of previous matched encodings in buffered frames
    prev_matched_ids = []
    time_dict = {}  # {id: leaving_time}
    last_detected_time = dt.datetime.min
    results = []  # [[closest_id, distance], ...]

    guest_count = 0
    user_names = []  # name of detected people in the current frame
    user_name_dict = {}  # {id: name}
    record_frame_count = 0

    video_capture = cv2.VideoCapture(1)
    for frame_count in itertools.count():
        _, frame = video_capture.read()

        # For every two frames, Skip one frame.
        if frame_count % 2 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = small_frame[:, :, ::-1]
            new_locations = face_recognition.face_locations(rgb_small_frame)
            if len(new_locations) > 0 or (
                dt.datetime.now() - last_detected_time
            ).total_seconds() > 2.0:
                locations = new_locations
                last_detected_time = dt.datetime.now()
                encodings = face_recognition.face_encodings(
                    rgb_small_frame, locations)
                matched_ids = []
                for encoding in encodings:
                    id, distance = find_closest(known_face_encodings, encoding)
                    if distance > tolerance and len(guest_encodings) > 0:
                        id2, distance2 = find_closest(
                            guest_encodings, encoding)
                        id2 += len(known_face_encodings)
                        if distance2 < distance:
                            id, distance = id2, distance2
                    matched_ids.append(id)
                start = 0 if len(prev_locations) < frame_buffer_size else 1
                prev_locations[0:] = prev_locations[start:] + [locations]
                prev_encodings[0:] = prev_encodings[start:] + [encodings]
                prev_matched_ids[0:] = prev_matched_ids[start:] + [matched_ids]
                record_frame_count += 1
                # Generate results every frame_buffer_size frames
                if (record_frame_count % frame_buffer_size
                        == frame_buffer_size - 1):
                    # Get the indices of encodings that have the most matching
                    indices_list = same_face_indices(
                        prev_encodings, prev_locations, tolerance)
                    results = []
                    for encoding, location, indices in zip(
                            encodings, locations, indices_list):
                        ids = [prev_matched_ids[row][col]
                               for row, col in indices]
                        id = Counter(ids).most_common(1)[0][0]
                        distance = face_recognition.face_distance(
                            (known_face_encodings + guest_encodings)[id: id+1],
                            encoding)[0]
                        results.append(
                            (id, distance)
                        )
                    user_names = []
                    now = datetime.now()
                    for i, (id, min_distance) in enumerate(results):
                        # Is new guest
                        if min_distance > tolerance:
                            user_name = f'訪客{guest_count + 1}'
                            id = len(known_face_encodings) + \
                                len(guest_encodings)
                            guest_encodings.append(encodings[i])
                            guest_count += 1
                        # Is in database
                        elif id < len(known_face_encodings):
                            user_name = user_profile_list[id][2]
                        # Is old guest
                        else:
                            user_name = user_name_dict[id]

                        # 若現在距離人臉最後偵測時間大於十秒，將資料寫進資料庫

                        if (now - time_dict.setdefault(
                            user_name, dt.datetime.min))\
                                .total_seconds() > 10.0:
                            Insert_Measure_Info(
                                database_name, [user_name, now])
                            data = fetch_newest_temperature_db(database_name)
                            Update_Measure_Info(database_name, data)
                            print('寫入')

                        print(user_name)
                        time_dict[user_name] = now
                        user_name_dict[id] = user_name
                        user_names.append(user_name)
        # TODO: Match locations with results
        result_frame = draw_results(
            frame, locations, results, user_names, tolerance, font)
        cv2.imshow('Video', result_frame)
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()

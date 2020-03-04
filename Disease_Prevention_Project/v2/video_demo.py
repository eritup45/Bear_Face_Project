import cv2
from datetime import datetime
from my_compare import get_encodings, find_closest
import face_recognition
import datetime as dt
from collections import Counter
import itertools
from Insert_Measure_Info import Insert_Measure_Info


def mean_of_face(prev_encodings, prev_locations):
    selected_bools = []
    for i, x in enumerate(prev_encodings):
        selected_bools.append([])
        for y in x:
            selected_bools[i].append(False)
    for last_encoding, last_location\
            in zip(prev_encodings[-1], prev_locations[-1]):
        for encodings, locations, bools\
                in zip(
                    prev_encodings[:-1],
                    prev_locations[:-1],
                    selected_bools[:-1]
                ):
                print('test')
            closest = None
            min_distance = 1.0
            for i in range(len(encodings)):
                distance = face_recognition.face_distance(
                    [encodings[i]], last_encoding
                )
                if distance < min_distance and distance < 0.35\
                        and bools[i] is False:
                    closest = i
                    min_distance = distance
            if closest is not None:
                bools[closest] = True


if __name__ == '__main__':
    database_name = './teacher.db'
    user_profile_list = get_encodings(database_name)
    known_face_encodings = [x[0] for x in user_profile_list]
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    prev_encoding = None
    buffer_frame_count = 12
    prev_locations = []
    prev_encodings = []
    time_dict = {}  # record {id: leaving_time}
    cnt = 0
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if frame_count % frame_buffer_size == frame_buffer_size - 1:
            # 取最符合的encoding
            indices_list = same_face_indices(prev_encodings, prev_locations)
            results = []
            for face_encoding in face_encodings:
                results.append(find_closest(
                    known_face_encodings, face_encoding))
                # if prev_face_encoding is not None:
                #     results.append(find_closest(
                #         [prev_face_encoding], face_encoding))
                # prev_face_encoding = face_encoding

        if cnt % 12 == 0 and cnt is not 0:
            # 對encoding取平均
            mean_of_face(prev_encodings, prev_locations)
            cnt = 0
        else:
            cnt = cnt + 1

        # For every two frames, Skip one frame.
        process_this_frame = not process_this_frame
        for (top, right, bottom, left), (id, min_distance) \
                in zip(face_locations, results):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 70),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame,
                str(f'{id}, Diff: {min_distance: > .4f}'),
                (left + 6, bottom - 42),
                font, 1.0, (255, 255, 255), 1)
            cv2.putText(
                frame,
                str(f' {"Not Matched" if min_distance > 0.35 else "Matched"}'),
                (left + 6, bottom - 6),
                font, 1.0, (255, 255, 255), 1)
            now = datetime.now()
            # 若判斷現在時間-人物最後偵測時間大於十秒，則判斷此人離場，將此人資料寫進資料庫
            if (now - time_dict.setdefault(user_profile_list[id][2], dt.datetime.min))\
                    .total_seconds() > 10.0:
                Insert_Measure_Info(database_name, [user_id, now])
                print(user_id)

            time_dict[user_id] = now

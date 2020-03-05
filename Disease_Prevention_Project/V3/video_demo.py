import cv2
from datetime import datetime
from my_compare import get_encodings, find_closest
import face_recognition
import datetime as dt
from collections import Counter
import itertools
from Insert_Measure_Info import Insert_Measure_Info


# Get the indices of faces in a list of frames which
#  corresponding to each face in the last frame.
def same_face_indices(prev_encodings, prev_locations):
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
                if distance < min_distance and distance < 0.35\
                        and bools[col] is False:
                    closest = col
                    min_distance = distance
            if closest is not None:
                bools[closest] = True
                indices.append((row, closest))
        indices.append((len(prev_encodings) - 1, i))
    return same_indices


def draw_results(locations, results, scale=2):
    for (top, right, bottom, left), (id, min_distance) \
            in zip(locations, results):
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

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


if __name__ == '__main__':
    database_name = './teacher.db'
    user_profile_list = get_encodings(database_name)
    known_face_encodings = [x[0] for x in user_profile_list]
    video_capture = cv2.VideoCapture(0)
    prev_encoding = None
    frame_buffer_size = 20
    prev_locations = []
    prev_encodings = []
    prev_matched_ids = []
    prev_distances = []
    time_dict = {}  # record {id: leaving_time}
    last_detected_time = dt.datetime.min
    frame_count = 0
    results = []
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
                distances = []
                for encoding in encodings:
                    id, distance = find_closest(known_face_encodings, encoding)
                    matched_ids.append(id)
                    distances.append(distance)
                start = 0 if len(prev_locations) < frame_buffer_size else 1
                prev_locations[0:] = prev_locations[start:] + [locations]
                prev_encodings[0:] = prev_encodings[start:] + [encodings]
                prev_matched_ids[0:] = prev_matched_ids[start:] + [matched_ids]
                prev_distances[0:] = prev_distances[start:] + [distances]

        if frame_count % frame_buffer_size == frame_buffer_size - 1:
            # 取最符合的encoding
            indices_list = same_face_indices(prev_encodings, prev_locations)
            results = []
            for encoding, location, indices in zip(
                    encodings, locations, indices_list):
                ids = [prev_matched_ids[row][col] for row, col in indices]
                id = Counter(ids).most_common(1)[0][0]
                distance = face_recognition.face_distance(
                    known_face_encodings[id: id+1], encoding)[0]
                results.append(
                    (id, distance)
                )

        draw_results(locations, results)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        for id, min_distance in results:
            now = datetime.now()
            if min_distance > 0.35:
                user_id = 'guest'
            else:
                user_id = user_profile_list[id][2]
               
            # 若判斷現在時間-人物最後偵測時間大於十秒，則判斷此人離場，將此人資料寫進資料庫
            if (now - time_dict.setdefault(
                user_id, dt.datetime.min))\
                    .total_seconds() > 10.0:
                Insert_Measure_Info(database_name, [user_id, now])
                print(user_id)

            time_dict[user_id] = now

import cv2
from datetime import datetime
from my_compare import get_encodings, find_closest
import face_recognition
import datetime as dt
from collections import Counter


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


if __name__ == '__main__':
    user_profile_list = get_encodings('teacher.db')
    known_face_encodings = list(map(lambda x: x[0], user_profile_list))
    # Read image
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    prev_encoding = None
    frame_buffer_size = 20
    prev_locations = []
    prev_encodings = []
    prev_matched_ids = []
    prev_distances = []
    time_dict = {}  # record {id: leaving_time}
    frame_count = 0
    results = []
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            # encode the frame of camera
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)
            matched_ids = []
            distances = []
            for encoding in face_encodings:
                id, distance = find_closest(known_face_encodings, encoding)
                matched_ids.append(id)
                distances.append(distance)
            if len(prev_locations) < frame_buffer_size:
                prev_locations.append(face_locations)
                prev_encodings.append(face_encodings)
                prev_matched_ids.append(matched_ids)
                prev_distances.append(distances)
            else:
                prev_locations[0:] = prev_locations[1:] + [face_locations]
                prev_encodings[0:] = prev_encodings[1:] + [face_encodings]
                prev_matched_ids[0:] = prev_matched_ids[1:] + [matched_ids]
                prev_distances[0:] = prev_distances[1:] + [distances]

        if frame_count % frame_buffer_size == frame_buffer_size - 1:
            print(frame_count)
            # if True:
            # 取最符合的encoding
            indices_list = same_face_indices(prev_encodings, prev_locations)
            results = []
            for face_encoding, face_location, indices in zip(
                    face_encodings, face_locations, indices_list):
                ids = list(map(
                    (lambda index: prev_matched_ids[index[0]][index[1]]),
                    indices
                ))
                id = Counter(ids).most_common(1)[0][0]
                distance = face_recognition.face_distance(
                    known_face_encodings[id: id+1], face_encoding)[0]
                results.append(
                    (id, distance)
                )
        frame_count += 1

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
            if (now - time_dict.setdefault(
                user_profile_list[id][2], dt.datetime.min))\
                    .total_seconds() > 10.0:
                pass
                # send to database
            time_dict[id] = now

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

import cv2
from datetime import datetime
from my_compare import get_encodings, find_closest
import face_recognition
import datetime as dt


if __name__ == '__main__':
    user_profile_list = get_encodings('teacher.db')
    known_face_encodings = list(map(lambda x: x[0], user_profile_list))
    # Read image
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    prev_encoding = None
    buffer_frame_count = 20
    prev_locations = []
    prev_encodings = []
    time_dict = {}  # record {id: leaving_time}
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            # encode the frame of camera
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)
                if len(prev_locations) < buffer_frame_count:
                prev_locations.append(face_locations)
                prev_encodings.append(face_encodings)
            else:
                prev_locations[0:] = prev_locations[1:] + face_locations
                prev_locations[0:] = prev_locations[1:] + face_locations
            results = []
            for face_encoding in face_encodings:
                results.append(find_closest(
                    known_face_encodings, face_encoding))
                # if prev_face_encoding is not None:
                #     results.append(find_closest(
                #         [prev_face_encoding], face_encoding))
                # prev_face_encoding = face_encoding

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
                pass
                # send to database
            time_dict[id] = now

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

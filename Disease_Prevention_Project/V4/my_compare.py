import face_recognition
import sqlite3
from encode import convert_list
from pathlib import Path


# Return:
# min_distance: smallest distance
# id: the id of the min_distance
def find_closest(list_of_face_encodings, unknown_face_encoding):
    distances = face_recognition.face_distance(
        list_of_face_encodings, unknown_face_encoding)
    min_distance = min(distances)
    id = distances.argmin()
    return id, min_distance


def get_encodings(database):
    user_profile_list = []
    conn = sqlite3.connect(str(Path(__file__).parent.joinpath(database)))
    c = conn.cursor()
    # 印出 User_Profile 中的資料，並以Number欄位排序
    for en_face, ID, Name in c.execute(
            'SELECT Encoded_face, ID, Name FROM User_Profile'):
        profile = []
        if convert_list(en_face) is not None:
            en_face = convert_list(en_face)
        profile = [en_face, ID, Name]

        user_profile_list.append(profile)

    conn.close()
    return user_profile_list
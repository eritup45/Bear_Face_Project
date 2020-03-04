import face_recognition
import sqlite3
from pathlib import Path
import os

from encode import read_image_name, adapt_list

dirname = os.path.dirname(__file__)


def write_one_face_in_db(database, img_name, user_p):
    conn = sqlite3.connect(os.path.join(dirname, database))
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS User_Profile (\
        Unit_id TEXT,\
            Unit_name TEXT,\
                ID TEXT PRIMARY KEY,\
                    Name TEXT,\
                        Title_name TEXT,\
                            Encoded_face BLOB)")

    image = face_recognition.load_image_file(img_name)
    # 查詢面部編碼
    list_of_face_encodings = face_recognition.face_encodings(image)
    if adapt_list(list_of_face_encodings) is None:
        print(f"Can\'t find face: {img_name}")
    else:
        # Because ID is PRIMARY KEY, just pick one photo.
        for i, data in enumerate(adapt_list(list_of_face_encodings)):
            if i >= 1:
                print('Find two faces in: ', img_name)
                break
            c.execute("INSERT INTO User_Profile (Encoded_face, Unit_id, Unit_name, ID, Name, Title_name) \
                VALUES ('{}', '{}', '{}', '{}', '{}', '{}')"
                      .format(data, user_p[0], user_p[1], user_p[2], user_p[3], user_p[4]))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    database = 'teacher.db'
    my = ''
    my_dir = Path(os.path.join(dirname, 'test3.jpg'))
    user_profile = [None, None, 'test3', '黃韋翔', 'Student']
    write_one_face_in_db(database, my_dir, user_profile)

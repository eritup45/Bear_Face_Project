import face_recognition
import sqlite3
from pathlib import Path

from database_utils import adapt_list

def Insert_one_face_in_db(database, img_name, user_p):
    conn = sqlite3.connect(database)
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
        return False
    else:
        # Because ID is PRIMARY KEY, just pick one photo. 
        for i, data in enumerate(adapt_list(list_of_face_encodings)):
            if i >= 1:
                print('Find two faces in: ', img_name)
                return False
            c.execute("INSERT INTO User_Profile (Encoded_face, Unit_id, Unit_name, ID, Name, Title_name) \
                VALUES ('{}', '{}', '{}', '{}', '{}', '{}')"\
                .format(data, user_p[0], user_p[1], user_p[2], user_p[3], user_p[4]))
    conn.commit()
    conn.close()
    return True


if __name__ == '__main__':
    database = './teacher.db'
    my = 'D:\大三\專題\防疫專案\data\教職員\picture\Q222562603.jpg'
    my_dir = Path(my)
    user_profile = [None, None, 'Q222562603', '許琇君', None]
    Insert_one_face_in_db(database, my_dir, user_profile)

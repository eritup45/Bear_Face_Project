import face_recognition
import numpy as np
from pathlib import Path
import sqlite3
import json

# Return
# num_classes: how many images' class
# image_name: images' name
def read_image_name(directory):
    num_classes = 0
    index = []
    image_name = []

    root_dir = Path(directory)
    # root_dir = Path(PATH_IMG).joinpath('data', 'caltech_faces')
    for i, _dir in enumerate(root_dir.glob('**/*')):
        for img in _dir.glob('*'):
            image_name.append(img)
            index.append(i)

        num_classes += 1

    return num_classes, index, image_name

# Convert np to json, in order to store in sqlite
def adapt_list(list_of_face_encodings):
    cvt_list_of_face_encodings = []
    # TODO:目前只存一張人臉
    for i in range(len(list_of_face_encodings)):
        cvt_list_of_face_encodings.append(json.dumps(list_of_face_encodings[i].tolist()))
        # cvt_list_of_face_encodings = json.dumps(list_of_face_encodings[i].tolist())

    # TODO:檢查db裡None是否被存成字串'None'
    # Can't find face
    if len(list_of_face_encodings) == 0:
        print("Can\'t find face")
        cvt_list_of_face_encodings = None
    return cvt_list_of_face_encodings

# Convert json to list
def convert_list(cvt_list_of_face_encodings):
    # my_list = []

    # Can't find face
    if cvt_list_of_face_encodings == 'None':
        return None

    # TODO:CHECK whether list
    # for i, data in enumerate(cvt_list_of_face_encodings):
    #     my_list.append(np.array(json.loads(data)))

    my_list = np.array(json.loads(cvt_list_of_face_encodings))
    return my_list

def print_database(database):
    # 連接資料庫(必寫)
    conn = sqlite3.connect(database)
    c = conn.cursor()

    # 印出 contacts 中的資料，並以Number欄位排序
    for num, en_face_list in c.execute('SELECT Number, Encoded_face FROM contacts'):
        en_face = []
        if convert_list(en_face_list) is not None:
            for i, data in enumerate(convert_list(en_face_list)):
                en_face.append(data)
        print(en_face)
        print(num)
        print('--------------------------------------')

    # # SElECT 查詢所有紀錄
    # SQL = "SELECT * FROM contacts"
    # c.execute(SQL)
    # print(c.fetchall())

    conn.close()

def write_face_encodings_in_db(directory, database):
    cnt = 0
    num_classes, index, image_name = read_image_name(directory)

    # 連接資料庫(必寫)
    conn = sqlite3.connect(database)
    c = conn.cursor()

    # 建立資料表 contacts
    # Number, name, student_ID, identity, encoded_face, date, temperature\
    c.execute("CREATE TABLE IF NOT EXISTS contacts (\
        Number INTEGER PRIMARY KEY AUTOINCREMENT,\
         Name TEXT,\
              Student_ID TEXT,\
                   Identity_card TEXT,\
                        Encoded_face BLOB,\
                             Date TEXT,\
                                  Temperature TEXT)")

    # 載入圖片
    for i, filename in enumerate(image_name):
        cnt = cnt + 1
        print(cnt)
        image = face_recognition.load_image_file(filename)

        # 查詢面部編碼
        list_of_face_encodings = face_recognition.face_encodings(image)

        # 在 contacts 中插入資料
        if adapt_list(list_of_face_encodings) is not None:
            for i, data in enumerate(adapt_list(list_of_face_encodings)):
                c.execute("INSERT INTO contacts (Encoded_face) VALUES ('{}')"
                        .format(data))

    # 寫入資料庫(必寫)
    conn.commit()
    conn.close()

def main():
    directory = 'D:\大三\專題\防疫專案\data\教職員'
    database = 'teacher.db'
    write_face_encodings_in_db(directory, database)
    print_database(database)

if __name__ == "__main__":
    main()

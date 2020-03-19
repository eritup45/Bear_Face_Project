import face_recognition
from pathlib import Path
import sqlite3
import json

# written by us
from Excel import read_excel
from utils import parse_args
from database_utils import adapt_list, convert_list

# Return
# num_classes: how many images' class
# image_name: images' name


def read_image_name(directory):
    num_classes = 0
    index = []
    image_name = []

    root_dir = Path(directory)
    for i, _dir in enumerate(root_dir.glob('**/*')):
        for img in _dir.glob('*'):
            image_name.append(img)
            index.append(i)

        num_classes += 1

    return num_classes, index, image_name


def print_database(database):
    # 連接資料庫(必寫)
    conn = sqlite3.connect(database)
    c = conn.cursor()

    # 印出 User_Profile 中的資料
    for id, en_face_list in c.execute('SELECT ID, Encoded_face FROM User_Profile'):
        en_face = []
        if convert_list(en_face_list) is not None:
            for i, data in enumerate(convert_list(en_face_list)):
                en_face.append(data)
        print(en_face)
        print(id)
        print('--------------------------------------')

    # # SElECT 查詢所有紀錄
    # SQL = "SELECT * FROM User_Profile"
    # c.execute(SQL)
    # print(c.fetchall())

    conn.close()


def write_face_encodings_in_db(database, img_directory, excel_file):
    cnt = 0
    # num_classes, index, image_name = read_image_name(img_directory)

    # 連接資料庫(必寫)
    conn = sqlite3.connect(database)
    c = conn.cursor()

    # 個人資料
    # unit_id: 學校單位, unit_name: 單位名稱, ID: 身分證, Name: 姓名, title_name: 職稱
    c.execute("CREATE TABLE IF NOT EXISTS User_Profile (\
        Unit_id TEXT,\
            Unit_name TEXT,\
                ID TEXT PRIMARY KEY,\
                    Name TEXT,\
                        Title_name TEXT,\
                            Encoded_face BLOB)")

    # 會變動的資料
    c.execute("CREATE TABLE IF NOT EXISTS Measure_Info (\
        Idx INTEGER PRIMARY KEY AUTOINCREMENT,\
            ID TEXT,\
                Date TEXT,\
                    Temp REAL,\
                        Status INTEGER)")

    # 載入圖片
    profile_list = read_excel(excel_file)
    for Unit_id, Unit_name, ID, Name, Title_name, _ in profile_list:
        filename = Path(img_directory).joinpath(ID + '.jpg')
        try:
            image = face_recognition.load_image_file(filename)
        except:
            print("Error loading: ", filename)
            continue

        # 查詢面部編碼
        list_of_face_encodings = face_recognition.face_encodings(
            image, num_jitters=10)
        # 在 User_Profile 中插入資料
        if adapt_list(list_of_face_encodings) is None:
            print(f"Can\'t find face: {filename}")
        else:
            # Because ID is PRIMARY KEY, just pick one photo.
            for i, data in enumerate(adapt_list(list_of_face_encodings)):
                if i >= 1:
                    print('Find two faces in: ', filename)
                    break
                c.execute("INSERT INTO User_Profile (Encoded_face, Unit_id, Unit_name, ID, Name, Title_name) \
                    VALUES ('{}', '{}', '{}', '{}', '{}', '{}')".format(data, Unit_id, Unit_name, ID, Name, Title_name))
                # print("INSERT succuess: ", filename)

    # 寫入資料庫(必寫)
    conn.commit()
    conn.close()

# Only accept Excel in form (Unit_id, Unit_name, ID, Name, Title_name, dist_type)
# Pictures' name should be "same" as ID.


def main():
    args = parse_args()
    database = args.db
    img_directory = args.image
    excel_file = args.excel

    write_face_encodings_in_db(database, img_directory, excel_file)
    # print_database(database)


if __name__ == "__main__":
    main()

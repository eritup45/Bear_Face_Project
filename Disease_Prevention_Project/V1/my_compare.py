import face_recognition
import sqlite3
import time
from encode import convert_list

def my_compare(list_of_face_encodings, unknown_face_encodings):
    for i, data in enumerate(list_of_face_encodings):
        if data is not None:
            results = face_recognition.compare_faces(
                [data], unknown_face_encodings, tolerance=0.56)

        # TODO:
        if results[0] == True:
            print("It's a picture of me!")
            print(i)
            break
        else:
            print("It's not a picture of me!")



def get_encodings(database):
    list_of_face_encodings = []
    conn = sqlite3.connect(database)
    c = conn.cursor()
    # 印出 contacts 中的資料，並以Number欄位排序
    for num, en_face_list in c.execute('SELECT Number, Encoded_face FROM contacts ORDER BY Number'):
        en_face = []
        if convert_list(en_face_list) is not None:
            for i, data in enumerate(convert_list(en_face_list)):
                en_face.append(data)
        # en_face = convert_list(en_face)
        list_of_face_encodings.append(en_face)

    conn.close()
    return list_of_face_encodings

if __name__ == '__main__':
    start = time.time()
    list_of_face_encodings = get_encodings('demo.db')

    # Read image 
    unknown_picture = face_recognition.load_image_file("data\\test123.JPG")
    unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

    my_compare(list_of_face_encodings, unknown_face_encoding)
    end = time.time()
    print('Time: ', start - end)
    

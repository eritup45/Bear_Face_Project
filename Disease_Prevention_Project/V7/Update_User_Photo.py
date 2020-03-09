import easygui
import cv2
from pathlib import Path
import os
import sqlite3
import sys

from Update_User_Profile import Update_one_face_in_db
from Insert_User_Profile import Insert_one_face_in_db

# Return path of photo and save in ./pictures
def take_photo(ID):
    cap = cv2.VideoCapture(0)
    while(1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(("./pictures/" + str(ID) + ".jpg"), frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return ("./pictures/" + str(ID) + ".jpg")

def main():
    database = 'teacher.db'
    conn = sqlite3.connect(database)
    c = conn.cursor()
    if not os.path.isdir('./pictures'):
        os.makedirs('./pictures')
    while(1):
        try:
            new_User = easygui.enterbox(msg="請輸入：\"姓名 身分證\" (中間請用空格隔開)\n"\
                ,title='新增/更新臉部資料').split()
            if len(new_User) != 2:
                print('Insert Error, please enter \"姓名 身分證\" again.')
                easygui.msgbox('Insert Error, please enter \"姓名 身分證\" again.')
                continue
        except:
            print('Insert Error, please enter \"姓名 身分證\" again.')
            easygui.msgbox('Insert Error, please enter \"姓名 身分證\" again.')
            continue

        ID = str(new_User[1])
        Name = str(new_User[0])
        user_profile = [None, None, ID, Name, None]

        data = ''
        photo_path = take_photo(ID)
        for data in c.execute("SELECT ID, Name FROM User_Profile Where ID = '{}'".format(ID)):
            if(Update_one_face_in_db(database, photo_path, user_profile)):
                easygui.msgbox('更新成功!')
                print('Update success!', ID, Name)
        if len(data) == 0:
            if(Insert_one_face_in_db(database, photo_path, user_profile)):
                easygui.msgbox('未找到符合身分，已加入資料庫')
                print('ID not found, Create new face.')
        
    conn.commit()
    conn.close()

if __name__ == '__main__':
    main()

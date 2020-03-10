
# cv2.imwrite(str(Path('./pictures/27.jpg')), frame)

# cv2.imwrite(str(Path(
# 'D:\\大三\\專題\\防疫專案\\Others\\Bear_Face_Project\\Disease_Prevention_Project\\V7/pictures/27.jpg')), frame)


import easygui
import cv2
from pathlib import Path
import os
import sqlite3
import sys

from Update_User_Profile import Update_one_face_in_db
from Insert_User_Profile import Insert_one_face_in_db

# Return 3 words in the back of ID


def five_last_ID(ID):
    return str(ID[7:10])

# Return path of photo and save in ./pictures


def take_photo(ID, pic_dir):
    cap = cv2.VideoCapture(int(sys.argv[2]))
    while(1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            status = cv2.imwrite(
                str(Path('D:\大三').joinpath(str(ID) + ".jpg")), frame)
            print('status: ', status)
            cv2.imwrite(str(Path(pic_dir).joinpath(str(ID) + ".jpg")), frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return str(Path(pic_dir).joinpath(str(ID) + ".jpg"))


def Update_User_Photo(database):
    print(str(Path(sys.argv[1])))
    root_dir = Path(sys.argv[1])
    database = str(root_dir.joinpath(database))
    pic_dir = str(root_dir.joinpath('./pictures'))

    conn = sqlite3.connect(database)
    c = conn.cursor()
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    while(1):
        try:
            new_User = easygui.enterbox(msg="請輸入：\"姓名 身分證後五碼\" (中間請用空格隔開)\n"\
                ,title='新增/更新臉部資料').split()
            if len(new_User) != 2:
                print('Insert Error, please enter \"姓名 身分證後五碼\" again.')
                easygui.msgbox('Insert Error, please enter \"姓名 身分證後五碼\" again.')
                continue
        except:
            print('Insert Error, please enter \"姓名 身分證後五碼\" again.')
            easygui.msgbox('Insert Error, please enter \"姓名 身分證後五碼\" again.')
            continue

        ID = str(new_User[1])
        Name = str(new_User[0])
        user_profile = [None, None, ID, Name, None]

        photo_path = take_photo(ID, pic_dir)
        data_list = []
        for data in c.execute("SELECT\
            Unit_id, Unit_name, ID, Name, Title_name\
                  FROM User_Profile Where Name = '{}'".format(Name)):
            # [[Unit_id, Unit_name, ID, Name, Title_name], ... ]
            data_list.append(data[0], data[1], data[2], data[3], data[4])
        # ID not found
        if len(data_list) == 0:
            if(Insert_one_face_in_db(database, photo_path, user_profile)):
                easygui.msgbox('未找到符合身分。加入資料庫')
            else:
                easygui.msgbox('臉部偵測錯誤，請重新檢測!')
        # Got the same name, find ID's last 5 words 
        elif len(data_list) > 1:
            # Find the same ID by last 5 words, and get the correct user_profile
            for tmp_Unit_id, tmp_Unit_name, tmp_ID, tmp_Name, tmp_Title_name in data_list:
                if five_last_ID(tmp_ID) == five_last_ID(ID):
                    user_profile = [tmp_Unit_id, tmp_Unit_name,
                                    tmp_ID, tmp_Name, tmp_Title_name]
                    break
            if(Update_one_face_in_db(database, photo_path, user_profile)):
                easygui.msgbox('更新成功!')
            else:
                easygui.msgbox('臉部偵測錯誤，請重新檢測!')
        # Normal case
        else:
            user_profile = data_list
            if(Update_one_face_in_db(database, photo_path, user_profile)):
                easygui.msgbox('更新成功!')
            else:
                easygui.msgbox('臉部偵測錯誤，請重新檢測!')

    conn.commit()
    conn.close()


if __name__ == '__main__':
    Update_User_Photo('./Release/teacher.db')

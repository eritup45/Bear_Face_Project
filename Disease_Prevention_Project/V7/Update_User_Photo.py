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
    return str(ID[-5:])

# Return path of photo and save in ./pictures
def save_photo(ID, pic_dir, frame):
    # Got same name
    if ID is not None:
        cv2.imencode('.jpg', frame)[1].tofile(str(Path(pic_dir).joinpath(str(ID) + ".jpg")))
        return str(Path(pic_dir).joinpath(str(ID) + ".jpg"))

# Update new face in databse 
def Update_User_Photo(database, frame):
    print(str(Path(sys.argv[1])))
    root_dir = Path(sys.argv[1])
    database = str(root_dir.joinpath(database))
    pic_dir = str(root_dir.joinpath('./pictures'))

    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)

    try:
        new_User = easygui.enterbox(msg="請輸入：\"姓名\"\n"\
            ,title='新增/更新臉部資料').split()
    except:
        easygui.msgbox('Insert Error, please enter \"姓名\" again.')
        return

    Name = str(new_User[0])
    # Only 5 words accepted
    
    conn = sqlite3.connect(database)
    c = conn.cursor()
    data_list = []
    for data in c.execute("SELECT\
        Unit_id, Unit_name, ID, Name, Title_name\
                FROM User_Profile Where Name = '{}'".format(Name)):
        # [[Unit_id, Unit_name, ID, Name, Title_name], ... ]
        data_list.append([ data[0], data[1], data[2], data[3], data[4] ])
    conn.commit()
    conn.close()
    # ID not found
    if len(data_list) == 0:
        easygui.msgbox('未找到符合身分。')
        # if(Insert_one_face_in_db(database, photo_path, user_profile)):
        #     easygui.msgbox('未找到符合身分。加入資料庫')
        # else:
        #     easygui.msgbox('臉部偵測錯誤，請重新檢測!')
    # Got the same name, find ID's last 5 words 
    elif len(data_list) > 1:
        try:
            ID = easygui.enterbox(
                msg="與他人撞名\n請輸入：\"身分證後五碼\"\n", title='新增/更新臉部資料')
            print('ID:', ID)
            if len(ID) != 5:
                easygui.msgbox('Insert Error, please enter \"身分證後五碼\" again.')
                return
        except:
            easygui.msgbox('Insert Error, please enter \"身分證後五碼\" again.')
            return

        # Find the same ID by last 5 words, and get the correct user_profile
        for tmp_Unit_id, tmp_Unit_name, tmp_ID, tmp_Name, tmp_Title_name in data_list:
            if five_last_ID(tmp_ID) == five_last_ID(ID):
                user_profile = [tmp_Unit_id, tmp_Unit_name,
                                tmp_ID, tmp_Name, tmp_Title_name]
                break
        photo_path = save_photo(user_profile[2], pic_dir, frame)
        if(Update_one_face_in_db(database, photo_path, user_profile)):
            easygui.msgbox('更新成功!')
        else:
            easygui.msgbox('臉部偵測錯誤，請重新檢測!')
    # Normal case
    else:
        user_profile = data_list[0]
        photo_path = save_photo(user_profile[2], pic_dir, frame)
        if(Update_one_face_in_db(database, photo_path, user_profile)):
            easygui.msgbox('更新成功!')
        else:
            easygui.msgbox('臉部偵測錯誤，請重新檢測!')



if __name__ == '__main__':
    video_num = 0
    video_capture = cv2.VideoCapture(video_num)
    _, frame = video_capture.read()
    Update_User_Photo('./Release/teacher.db', frame)

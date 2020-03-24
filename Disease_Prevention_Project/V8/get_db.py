import sqlite3
from database_utils import convert_list
from pathlib import Path
from get_file_path import get_file_path


def get_user_profiles(database):
    user_profile_list = []
    print(str(Path(get_file_path()).joinpath(database)))
    conn = sqlite3.connect(str(Path(get_file_path()).joinpath(database)))
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

import sqlite3
import os
from pathlib import Path
from get_file_path import get_file_path


def Insert_Measure_Info(database, info):
    dirname = str(Path(get_file_path()).joinpath(database))
    ID = str(info[0])
    Date = str(info[1])
    conn = sqlite3.connect(dirname)
    c = conn.cursor()
    c.execute("INSERT INTO Measure_Info (ID, Date) \
                    VALUES ('{}', '{}')".format(ID, Date))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    database = 'teacher.db'
    info = ['A123456789', '2020-03-04 17:31:18.587212']
    Insert_Measure_Info(database, info)

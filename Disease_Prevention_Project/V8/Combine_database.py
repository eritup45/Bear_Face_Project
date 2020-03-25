import sqlite3
import os
from pathlib import Path
from get_file_path import get_file_path

def fetch_newest_temperature_db(database):
    conn = sqlite3.connect(str(
        Path(get_file_path()).joinpath(database)))
    c = conn.cursor()
    # Method 1
    # fetch the newset Date
    for data in c.execute('SELECT MAX(Date), Temp, Status, Temp_Index FROM newest_temperature'):
        pass
        # print(data)

    # Method 2
    # for Date, Temp, Status in c.execute('SELECT * FROM newest_temperature ORDER BY Date DESC LIMIT 1'):
    #     print(Date, Temp, Status)
    conn.commit()
    conn.close()
    return data

# Ordered by Idx, Update Date, Temp, Status in Table Measure_Info
# data: [Date, Temp, Status]


def Update_Measure_Info(database, data):
    conn = sqlite3.connect(str(
        Path(get_file_path()).joinpath(database)))
    Date = str(data[1])
    Temp = float(data[2])
    Status = int(data[3])
    Temp_Index = int(data[0])
    c = conn.cursor()
    c.execute("UPDATE Measure_Info\
        SET Temp_Index = '{}', Date = '{}', Temp = '{}', Status = '{}'\
        WHERE Idx = (SELECT MAX(Idx) from Measure_Info)".format(Temp_Index, Date, Temp, Status))
    conn.commit()
    conn.close()

# Just for testing
# Insert Date, Temp, Status in Table newest_temperature
# data: [[Date, Temp, Status], ...]


def Insert_newest_temperature(database, data):
    conn = sqlite3.connect(str(
        Path(get_file_path()).joinpath(database)))
    for Date, Temp, Status in data:
        c = conn.cursor()
        c.execute("INSERT INTO newest_temperature (Date, Temp, Status) \
                        VALUES ('{}', '{}', '{}')".format(Date, Temp, Status))
    conn.commit()
    conn.close()


def main():
    database = 'teacher.db'
    data = fetch_newest_temperature_db(database)
    Update_Measure_Info(database, data)
    # data = [['2020-03-06 10:23:48.279110', 37.2, 1],
    #         ['2020-03-06 10:29:31.058381', 36.5, 0]]
    # Insert_newest_temperature(database,data)


if __name__ == '__main__':
    main()

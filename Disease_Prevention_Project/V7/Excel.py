import xlrd


def read_excel(filename):
    data = xlrd.open_workbook(filename)  # 打开文件

    sheet1 = data.sheet_by_index(0)  # 通过索引获取表格

    profile_list = []

    for i in range(1, len(sheet1.col_values(0))):
        # print (sheet1.row_values(i)) #列印每一行的內容
        profile_list.append(sheet1.row_values(i))

    return profile_list


if __name__ == '__main__':
    cnt = 0
    profile_list = read_excel("data\\教職員\\現職人員及單位_1090227.xls")
    for Unit_id, Unit_name, ID, Name, Title_name, _ in profile_list:
        print(ID)
        cnt = cnt + 1

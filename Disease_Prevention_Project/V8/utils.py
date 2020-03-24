import argparse


def parse_args():
    desc = 'Set Path.\n'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-pe', '--excel', help='path to excel',
                        default=".\data\教職員\現職人員及單位_1090227.xls")
    parser.add_argument('-pi', '--image', help='path to image',
                        default=".\data\教職員\picture")
    parser.add_argument(
        '-db', '--db', help='path to database', default=".\\teacher.db")
    return parser.parse_args()

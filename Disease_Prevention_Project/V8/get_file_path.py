from pathlib import Path
import sys
# import argparse
# from parser import parser_args

# def parse_args():
#     parser.add_argument('--dist', type=str, required=True)
#     parser.add_argument('--camera', type=int, default=0)
#     return parser.parse_args()


def get_file_path():
    if len(sys.argv) >= 2:
        return sys.argv[1]
    # Use as debug
    else:
        return str(Path(__file__).parent)
    # try:
    #     this_file = __file__
    # except NameError:
    #     this_file = sys.argv[1]
    # this_file = op.abspath(this_file)
    # if getattr(sys, 'frozen', False):
    #     return getattr(sys, '_MEIPASS', op.dirname(sys.executable))
    # else:
    #     return os.path.dirname(this_file)

# def get_file_path():
#     if getattr(sys, 'frozen', False):
#         # If the application is run as a bundle, the pyInstaller bootloader
#         # extends the sys module by a flag frozen=True and sets the app
#         # path into variable _MEIPASS'.
#         return sys._MEIPASS
#     else:
#         return os.path.dirname(os.path.abspath(__file__))

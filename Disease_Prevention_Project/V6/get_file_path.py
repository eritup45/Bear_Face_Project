import os
import sys
import os.path as op
def get_file_path():
    # print('argv[1]:', sys.argv[1])
    return sys.argv[1]
    # try:
    #     this_file = __file__
    # except NameError:
    #     this_file = sys.argv[1]
    # this_file = op.abspath(this_file)
    # if getattr(sys, 'frozen', False):
    #     return getattr(sys, '_MEIPASS', op.dirname(sys.executable))
    # else:
    #     return os.path.dirname(this_file)

# import sys, os

# def get_file_path():
#     if getattr(sys, 'frozen', False):
#         # If the application is run as a bundle, the pyInstaller bootloader
#         # extends the sys module by a flag frozen=True and sets the app 
#         # path into variable _MEIPASS'.
#         return sys._MEIPASS
#     else:
#         return os.path.dirname(os.path.abspath(__file__))

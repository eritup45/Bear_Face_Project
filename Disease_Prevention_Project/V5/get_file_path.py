import sys, os

def get_file_path():
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the pyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app 
        # path into variable _MEIPASS'.
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

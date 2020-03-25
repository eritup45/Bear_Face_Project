import cv2
import easygui
import os
import camera
from pygrabber.dshow_graph import FilterGraph


def set_cam():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    filepath = "setting.txt"
    if os.path.isfile(filepath):
        print("setting.txt存在。")
        f = open('setting.txt', 'r')
        device = f.read()
        return devices.index(device)
    else:
        print("setting.txt不存在。")
    for i in range(len(devices)):
        main_camera = camera.Camera(i)
        frame = main_camera.read_bgr()
        cv2.imshow('frame', frame)
        value = easygui.ynbox("Is this the video camera?")
        if value is True:
            fp = open("setting.txt", "a")
            fp.write(str(devices[i]))
            fp.close()
            main_camera.stop()
            cv2.destroyAllWindows()
            return i
        else:
            continue
    return None


if __name__ == "__main__":
    set_cam()

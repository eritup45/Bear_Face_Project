from pygrabber.dshow_graph import FilterGraph
import threading
import numpy as np


class Camera:
    def __init__(self, device_id):
        self.graph = FilterGraph()
        self.graph.add_video_input_device(device_id)
        self.graph.add_sample_grabber(self.img_cb)
        self.graph.add_null_render()
        self.graph.prepare_preview_graph()
        self.graph.run()

        self.image_grabbed = None
        self.image_done = threading.Event()

    def img_cb(self, image):
        self.image_grabbed = np.flip(image, 2)
        self.image_done.set()

    def read(self):
        self.graph.grab_frame()
        self.image_done.wait(1000)
        return self.image_grabbed

    def read_bgr(self):
        self.read()
        return np.flip(self.read(), 2)

    def stop(self):
        self.graph.stop()
        self.image_done.clear()

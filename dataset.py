import chainer
import cv2
import pathlib


class FaceData(chainer.dataset.DatasetMixin):

    def __init__(self):
        path = pathlib.Path("")

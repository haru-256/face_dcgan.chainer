import chainer
import cv2
import pathlib


class FaceData(chainer.dataset.DatasetMixin):

    def __init__(self):
        data_path = pathlib.Path("data")
        abs_data_path=data_path.resolve()
        print("data path:", abs_data_path)
        for img_path in data_path.glob(*/*.jpg):
            img = cv2.imread()

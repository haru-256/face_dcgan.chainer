import chainer
import cv2
import pathlib
import numpy as np
"""
ChainerのDatasetは
初期化 __init__
内部に持っているデータの数を返却する __len__(self)
このメソッドは整数でデータ数を返却する
i番目のデータを取得する get_example(self, i)
を実装する
このメソッドは、return image_array, labelの様なことをする
"""


class FaceData(chainer.dataset.DatasetMixin):
    """
        cropped_data_128ディレクトリからデータを読み込む
    """

    def __init__(self):
        data_path = pathlib.Path("cropped_data_128")
        abs_data_path = data_path.resolve()
        print("data path:", abs_data_path)

        # Reading data
        self.data = [
            np.float32(
                cv2.cvtColor(cv2.imread(str(img_path)),
                             cv2.COLOR_BGR2RGB).transpose(-1, 0, 1))
            for img_path in abs_data_path.glob("*/*.jpg")
        ]

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        return self.data[i]

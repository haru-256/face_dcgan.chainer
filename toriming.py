import cv2
import pathlib
import matplotlib.pyplot as plt

data_path = pathlib.Path("../data/lfw/")
abs_data_path = data_path.resolve()
save_path = pathlib.Path(".") / "cropped_data_100"
abs_save_path = save_path.resolve()

if not abs_save_path.exists():
    abs_save_path.mkdir()

print("save dir:", abs_save_path)

# center crop 150*150
dx, dy = 100, 100
for i, img_path in enumerate(abs_data_path.glob("*/*.jpg")):
    img = cv2.imread(str(img_path))
    cropped_img = img[125 - dx // 2:125 + dx // 2, 125 - dy // 2:125 + dy // 2]
    # save
    tmp = pathlib.Path(str(abs_save_path / img_path.parts[-2]))
    if not tmp.exists():
        tmp.mkdir()
    cv2.imwrite(
        str(abs_save_path / "/".join(img_path.parts[-2:])), cropped_img)

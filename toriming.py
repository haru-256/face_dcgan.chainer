import cv2
import pathlib
import matplotlib.pyplot as plt

data_path = pathlib.Path("Yukio_Hatoyama_0001.jpg")
abs_data_path = data_path.resolve()
save_path = abs_data_path.parent / "cropped_data/"

if not save_path.exists():
    save_path.mkdir()

print("save_path:", save_path)

img = cv2.imread(str(data_path))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# center crop 150*150
dx, dy = 150, 150
cropped_img = img[125 - dx // 2:125 + dx // 2, 125 - dy // 2:125 + dy // 2]
plt.imshow(cropped_img)
plt.show()

# save
cv2.imwrite(str(save_path /data_path), cropped_img)

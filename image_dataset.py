# %%
from chainer.datasets import ImageDataset
import pathlib

data_dir = pathlib.Path("cropped_data/")
data_path = []
for path in data_dir.glob("*.jpg"):
    data_path.append(path)
print(data_path)
data = ImageDataset(paths=data_path)
print(data)
image = data.get_example(0)
print(data.get_example(0).shape)

# %%
import matplotlib.pyplot as plt
plt.imshow(image.transpose(1, 2, 0).astype("int"))
print(data[0].shape)
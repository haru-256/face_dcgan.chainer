import chainer
from chainer import Variable
from chainer import serializers
from generator import Generator
from visualize import combine_images
import pathlib
import matplotlib.pyplot as plt

path = pathlib.Path("result3_a/gen_iter_31015.npz")
abs_path = path.resolve()

gen = Generator()  # prepare model
serializers.load_npz(path, gen)  # load pretraining model
# serializers.load_npz(path/"gen_iter_1406250.npz", gen) # load pretraining model
xp = gen.xp  # get numpy or cupy

xp.random.seed(144)
z = Variable(xp.random.uniform(
    -1, 1, (10 * 10, 100)).astype("f"))  # make noize whose elements are all

with chainer.using_config('train', False):
    x = gen(z)
x = chainer.backends.cuda.to_cpu(x.data)  # send data to cpu

x = (x * 127.5 + 127.5) / 255
x = x.transpose(0, 2, 3, 1)  # NCHW->NHWCに変形
x = combine_images(x)
plt.figure(figsize=(10, 10))
plt.imshow(x)
plt.axis("off")
plt.savefig("hoge.png")
plt.show()

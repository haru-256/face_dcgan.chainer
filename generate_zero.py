import chainer
from chainer import Variable
from chainer import serializers
from generator import Generator
from visualize import combine_images
import pathlib
import matplotlib.pyplot as plt
import argparse

# パーサーを作る
parser = argparse.ArgumentParser(
    prog='animation.py',  # プログラム名
    usage='Make animation of generated images by GIF',  # プログラムの利用方法
    description='description',  # 引数のヘルプの前に表示
    epilog='end',  # 引数のヘルプの後で表示
    add_help=True,  # -h/–help オプションの追加
)

# 引数の追加
parser.add_argument('-s', '--seed', help='seed',
                    type=int, required=True)
parser.add_argument('-n', '--number', help='the number of experiments',
                    type=int, required=True)
parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                    action='version',
                    default=False)
parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 300',
                    type=int, default=300)

# 引数を解析する
args = parser.parse_args()
number = args.number  # nmber of experiments
seed = args.seed  # seed
strings = "{0}_{1}".format(number, seed)
epoch = args.epoch
path = pathlib.Path(
    "./result_{0}/result_{1}/gen_epoch_{2}.npz".format(number, strings, epoch))
abs_path = path.resolve()

gen = Generator()  # prepare model
serializers.load_npz(path, gen)  # load pretraining model
# serializers.load_npz(path/"gen_iter_1406250.npz", gen) # load pretraining model
xp = gen.xp  # get numpy or cupy

xp.random.seed(seed)
z = Variable(xp.random.uniform(
    -1, 1, (10 * 10, 100)).astype("f"))  # make noize whose elements are all

with chainer.using_config('train', False):
    with chainer.using_config('enable_backprop', False):
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

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.backends.cuda
from chainer import Variable


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(np.sqrt(total))
    rows = int(np.ceil(float(total) / cols))
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros(
        (height * rows, width * cols, 3), dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        for ch in range(3):
            combined_image[width*i:width*(i+1), height*j:height*(j+1), ch] =\
                image[:, :, ch]
    return combined_image


def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        n_images = rows * cols

        xp = gen.xp  # get module
        xp.random.seed(seed)  # fix seed
        np.random.seed(seed)  # fix seed
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        # test, evaluationの時は以下の２つを設定しなければならない
        # https://qiita.com/mitmul/items/1e35fba085eb07a92560
        # 'train'をFalseにすることで，train時とテスト時で挙動が異なるlayer(BN, Dropout)
        # を制御する
        with chainer.using_config('train', False):
            # 'enable_backprop'をFalseとすることで，無駄な計算グラフの構築を行わない
            # ようにしメモリの消費量を抑える.
            with chainer.using_config('enable_backprop', False):
                x = gen(z)
        x = chainer.backends.cuda.to_cpu(x.data)
        xp.random.seed()
        np.random.seed()

        x = (x * 127.5 + 127.5) / 255  # 0~255に戻し0~1へ変形
        x = x.transpose(0, 2, 3, 1)  # NCHW->NHWCに変形
        x = combine_images(x)
        plt.imshow(x)
        plt.axis("off")
        preview_dir = pathlib.Path('{}/preview'.format(dst))
        preview_path = preview_dir /\
            'image_{:}epoch.png'.format(trainer.updater.epoch)
        if not preview_dir.exists():
            preview_dir.mkdir()
        plt.tight_layout()
        plt.savefig(preview_path)

    return make_image

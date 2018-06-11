import chainer
from chainer import training
from chainer.training import extensions
from chainer.datasets import ImageDataset
from chainer.serializers import save_npz
from dataset import FaceData

# from discriminator import Discriminator  # Dence nobias
from discriminator_md import Discriminator  # GAP nobias
from generator import Generator
from updater import DCGANUpdater
from visualize import out_generated_image
# from accuracy_reporter import accuracy_report
import pathlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def make_optimizer(model, alpha=0.0002, beta1=0.5):
    """
    Setup an optimizer
    """
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0001), 'hook_dec')

    return optimizer


if __name__ == '__main__':
    import numpy as np
    import argparse

    # パーサーを作る
    parser = argparse.ArgumentParser(
        prog='train',  # プログラム名
        usage='train DCGAN',  # プログラムの利用方法
        description='description',  # 引数のヘルプの前に表示
        epilog='end',  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )

    # 引数の追加
    parser.add_argument('-s', '--seed', help='seed',
                        type=int, required=True)
    parser.add_argument('-n', '--number', help='the number of experiments.',
                        type=int, required=True)
    parser.add_argument('--hidden', help='the number of codes of Generator.',
                        type=int, default=100)
    parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 300',
                        type=int, default=300)
    parser.add_argument('-bs', '--batch_size', help='batch size. defalut value is 128',
                        type=int, default=128)
    parser.add_argument('-g', '--gpu', help='specify gpu by this number. defalut value is 0',
                        choices=[0, 1], type=int, default=0)
    parser.add_argument('-dis', '--discriminator',
                        help='specify discriminator by this number. any of following;'
                        ' 0: original, 1: minibatch discriminatio, 2: feature matching. defalut value is 0',
                        choices=[0, 1, 2], type=int, default=0)
    parser.add_argument('-ts', '--tensor_shape',
                        help='specify Tensor shape by this numbers. first args denotes to B, seconds to C.'
                        ' defalut value are B:32, C:8',
                        type=int, default=[32, 8], nargs=2)
    parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                        action='version',
                        default=False)

    # 引数を解析する
    args = parser.parse_args()

    gpu = args.gpu
    batch_size = args.batch_size
    n_hidden = args.hidden
    epoch = args.epoch
    seed = args.seed
    number = args.number  # number of experiments
    out = "result_{0}/result_{0}_{1}".format(number, seed)

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# n_hidden: {}'.format(n_hidden))
    print('# epoch: {}'.format(epoch))
    print('# out: {}'.format(out))
    print('# seed: {}'.format(seed))
    print('# Tensor shape is A x {0} x {1}'.format(
        args.tensor_shape[0], args.tensor_shape[1]))

# import discrimination
    if args.discriminator == 0:
        print("# Original Discriminator")
        from discriminator import Discriminator
        from updater import DCGANUpdater
    elif args.discriminator == 1:
        print("# Discriminator applied Minibatch Discrimination")
        from discriminator_md import Discriminator
        from updater import DCGANUpdater
    """
    elif args.discriminator == 2:
        print("# Discriminator applied matching")
        from discriminator_fm import Discriminator
        from updater_fm import DCGANUpdater
    """
    print('')

    # fix seed
    np.random.seed(seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)

    # Set up a neural network to train
    gen = Generator(n_hidden=n_hidden)
    dis = Discriminator(B=args.tensor_shape[0], C=args.tensor_shape[1])

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # Prepare Dataset
    """
    train = FaceData()
    """
    data_dir = pathlib.Path("./cropped_data_128_df")
    abs_data_dir = data_dir.resolve()
    print("data dir path:", abs_data_dir)
    data_path = [path for path in abs_data_dir.glob("*/*.jpg")]
    print("data length:", len(data_path))
    data = ImageDataset(paths=data_path)  # dtype=np.float32
    train_iter = chainer.iterators.SerialIterator(data, batch_size)

    # Set up a updater and trainer
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen,
            'dis': opt_dis
        },
        device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    snapshot_interval = (10, 'epoch')
    display_interval = (1, 'epoch')
    # storage method is hdf5
    trainer.extend(
        extensions.snapshot(
            filename='snapshot_epoch_{.updater.epoch}.npz',
            savefun=save_npz),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(
            gen, 'gen_epoch_{.updater.epoch}.npz', savefun=save_npz),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(
            dis, 'dis_epoch_{.updater.epoch}.npz', savefun=save_npz),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport([
            'epoch', 'iteration', 'gen/loss', 'dis/loss', 'elapsed_time',
        ]),
        trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=20))
    trainer.extend(
        out_generated_image(gen, dis, 7, 7, seed, out),
        trigger=display_interval)
    trainer.extend(
        extensions.PlotReport(
            ['gen/loss', 'dis/loss'],
            x_key='epoch',
            file_name='loss_{0}_{1}.jpg'.format(number, seed),
            grid=False))
    trainer.extend(extensions.dump_graph("gen/loss", out_name="gen.dot"))
    trainer.extend(extensions.dump_graph("dis/loss", out_name="dis.dot"))

    # Run the training
    trainer.run()

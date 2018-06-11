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


def main():
    import numpy as np
    # fix seed
    seed = 0
    np.random.seed(seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)

    gpu = 0  # GAP: 0, Dense: 1
    batch_size = 128
    n_hidden = 100
    epoch = 300  # Dence:100 GAP:300
    number = 6  # number of experiments
    out = "result_{0}_{1}".format(number, seed)

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# n_hidden: {}'.format(n_hidden))
    print('# epoch: {}'.format(epoch))
    print('# out: {}'.format(out))
    print('# seed: {}'.format(seed))
    print('')

    # Set up a neural network to train
    gen = Generator()
    dis = Discriminator(B=32, C=8)

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


if __name__ == '__main__':
    main()

import chainer
from chainer import training
from chainer.training import extensions
from dataset import FaceData

from discriminator import Discriminator
from generator import Generator
from updater import DCGANUpdater
from visualize import out_generated_image


# Setup an optimizer
def make_optimizer(model, alpha=0.0002, beta1=0.5):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    """
    optimizer.add_hook(
        chainer.optimizer_hooks.WeightDecay(0.0001), 'hook_dec')
    """
    return optimizer


def main():
    import numpy
    numpy.random.seed(0)
    import chainer
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(0)
    gpu = 0
    batch_size = 128
    n_hidden = 100
    epoch = 100
    seed = 0
    out = "result"

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# n_hidden: {}'.format(n_hidden))
    print('# epoch: {}'.format(epoch))
    print('')

    # Set up a neural network to train
    gen = Generator()
    dis = Discriminator()

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # Load the mnist dataset and make iterator
    train = FaceData()
    train_iter = chainer.iterators.SerialIterator(train, batch_size)

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
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(gen, 'gen_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(dis, 'dis_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'gen/loss', 'dis/loss', 'elapsed_time']),
        trigger=display_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(
        out_generated_image(gen, dis, 5, 5, seed, out),
        trigger=display_interval)
    trainer.extend(
        extensions.PlotReport(
            ['gen/loss', 'dis/loss'], x_key='epoch', file_name='loss.png'))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
from utils import Gamma_initializer


class Critic(chainer.Chain):
    """
    Critic

    build Critic.

    Parameters
    ----------------

    ksize: int
        kernel size. 4 or 5

    pad: int
        padding size. if ksize is 4, then pad has to be 1.
        if ksize is 5, then pad has to be 2.

    nobias: boolean
        whether don't apply bias to convolution layer with no BN layer.
    """

    def __init__(self, ksize=4, pad=1, ch=1024, nobias=True):
        super(Critic, self).__init__()
        print("Critic")
        with self.init_scope():
            # initializers
            conv_init = chainer.initializers.Normal(scale=0.02)
            gamma_init = Gamma_initializer(mean=1.0, scale=0.02)
            beta_init = chainer.initializers.Zero()

            # registar layers with variable
            self.c0 = L.Convolution2D(
                in_channels=None,
                out_channels=ch//16,
                ksize=ksize,
                pad=pad,
                stride=2,
                initialW=conv_init,
                nobias=nobias
            )  # (, 64, 64, 64)
            self.c1 = L.Convolution2D(
                in_channels=None,
                out_channels=ch//8,
                ksize=ksize,
                pad=pad,
                stride=2,
                initialW=conv_init,
                nobias=True
            )  # (, 128, 32, 32)
            self.c2 = L.Convolution2D(
                in_channels=None,
                out_channels=ch//4,
                ksize=ksize,
                pad=pad,
                stride=2,
                initialW=conv_init,
                nobias=True
            )  # (, 256, 16, 16)
            self.c3 = L.Convolution2D(
                in_channels=None,
                out_channels=ch//2,
                ksize=ksize,
                pad=pad,
                stride=2,
                initialW=conv_init,
                nobias=True
            )  # (, 512, 8,8)
            self.c4 = L.Convolution2D(
                in_channels=None,
                out_channels=ch,
                ksize=ksize,
                pad=pad,
                stride=2,
                initialW=conv_init,
                nobias=nobias
            )  # (, 1024, 4, 4)
            self.c5 = L.Convolution2D(
                in_channels=None,
                out_channels=1,
                ksize=ksize,
                pad=0,
                stride=1,
                initialW=conv_init,
                nobias=nobias
            )  # (, 1, 1, 1)

            self.bn1 = L.BatchNormalization(
                size=ch//8,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05
            )
            self.bn2 = L.BatchNormalization(
                size=ch//4,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05
            )
            self.bn3 = L.BatchNormalization(
                size=ch//2,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05
            )
            self.bn4 = L.BatchNormalization(
                size=ch,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05
            )

    def __call__(self, x):
        """
        Function that computes forward

        Parametors
        ----------------
        x: Variable
           input image data. this shape is (N, C, H, W)
        """
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = F.leaky_relu(self.bn4(self.c4(h)))
        y = self.c5(h)

        return y


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable
    import numpy as np

    z = np.random.normal(0, 1, (10, 3, 128, 128)).astype("f")
    model = Critic()
    img = model(Variable(z))

    # print(img)
    g = c.build_computational_graph(img)
    with open('critic_graph.dot', 'w') as o:
        o.write(g.dump())

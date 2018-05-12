import chainer
from chainer import Variable


class Accuracy_Reporter(chainer.Chain):
    """
    The class that reports accuracy of Discriminator

    Parameters
    ------------
    predictor: Link
       Discriminators

    report_name: str
       this report is whether fake image or real image


    """

    def __init__(self, predictor, report_name, n_image=100, seed=0):


def accuracy_report(gen, dis, data, n_images=100, seed=0, report_name="accuracy"):
    """
    This function measures accuracy of Discriminator.

    Parameters
    ---------------
    gen: chainer.Link
       Generator

    dis: chainer.link
       Discriminator

    data: chainer.Dataset
        Datasets that has real image

    n_image: int
        number of datas

    seed: int
        seed to fix result

    report_name: str
        report name used to pass the reporter
    """
    @chainer.training.make_extension()
    def cal_accuracy(trainer):
        xp = gen.xp  # get module
        xp.random.seed(seed)  # fix seed

        """prepare real image"""
        """
        real_image = data[xp.random.randint(
            len(data), size=n_images)]  # get real image
        real_image = chainer.backends.cuda.to_gpu(real_image.data)  # copy to gpu: cupy
        """

        """prepare fake image"""
        z = Variable(xp.asarray(gen.make_hidden(n_images)))  # get noize: cupy
        xp.random.seed()  # Free seed

        # test, evaluationの時は以下の２つを設定しなければならない
        # https://qiita.com/mitmul/items/1e35fba085eb07a92560
        # 'train'をFalseにすることで，train時とテスト時で挙動が異なるlayer(BN, Dropout)
        # を制御する
        with chainer.using_config('train', False):
            # 'enable_backprop'をFalseとすることで，無駄な計算グラフの構築を行わない
            # ようにしメモリの消費量を抑える.
            with chainer.using_config('enable_backprop', False):
                fake_image = gen(z)   
                # evaluate Discriminator
                # real = dis(real_image)
                fake = dis(fake_image)

        """
        real = chainer.backends.cuda.to_cpu(
                    real.data)  # copy x to cpu & get fake image: numpy
        """
        fake = chainer.backends.cuda.to_cpu(
                    fake.data)  # copy fake to cpu & get fake image: numpy
        # calculate accuracy
        sum = len(fake[fake > 0.5])  # 偽物を本物と正答した数
        # sum += len(real[real >= 0.5])  # 本物を本物と正答した数        
        # accuracy = sum / (len(real) + len(fake))
        accuracy = sum / len(fake)  # 謝り識別率

        # chainer.report({'accuracy': accuracy}, dis)
        print("generated image accuracy: ", accuracy)

    return cal_accuracy

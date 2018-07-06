from torchloop.dataset import torchvision_wrappers as tvw
from torchloop.util import cv_helper, exec_utils, tl_logging
logger=tl_logging.tl_logger(tl_logging.DEBUG, True)
import unittest

def gen_dataset():
    data_dir = '/home/bigtree/PycharmProjects/torchloop/data/'
    cifar10_loader = tvw.cifar10_loader(root_dir=data_dir, 
            batch_size=4, n_workers=2)
    train_loader_ = cifar10_loader.train_loader
    test_loader_ = cifar10_loader.test_loader
    classes = cifar10_loader.labels
    return train_loader_, test_loader_, classes

def gen_mnist_dataset():
    data_dir = '/home/bigtree/PycharmProjects/torchloop/data/mnist'
    mnist_loader = tvw.mnist_loader(root_dir=data_dir, 
            batch_size=4, n_workers=2)
    train_loader_ = mnist_loader.train_loader
    test_loader_ = mnist_loader.test_loader
    classes = mnist_loader.labels
    n_iter = mnist_loader.n_iter_per_epoch
    return train_loader_, test_loader_, classes, n_iter

class test(unittest.TestCase):
    def test_cifar10_loader(self):
        train_loader_, test_loader_, classes = \
                gen_dataset()
        dataiter = iter(train_loader_)
        images, labels = dataiter.next()
        print("sample size of the training loader mnist {}".\
                format(len(train_loader_)))
        print("sample size of the training loader mnist {}".\
                format(len(test_loader_)))
        cv_helper.im_show_plt(cv_helper.make_img_grid(images))
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
        cv_helper.im_toggle_show_plt()

    def test_plotter(self):
        train_loader_, test_loader_, classes = \
                gen_dataset()
        dataiter = iter(train_loader_)
        images, labels = dataiter.next()
        ######
        # log out something about images and labels
        logger.debug("images tensor {}, labels tensor {}".format(
            type(images), type(labels)))
        logger.debug("images size {}, labels size {}".format(
            images.size(), labels.size()))
        logger.debug("-----------------separater-------------------")

        logger.debug(' '.join(
            '%5s' % classes[labels[j]] for j in range(4)))
        cv_helper.plt_plotter.img_to_show(images)
        cv_helper.plt_plotter.toggle_show()
        #####
        # sleep for a few seconds 
        #####
        logger.debug("sleeping for a few seconds")
        exec_utils.sleep(1.5)
        
    def test_mnist_loader(self):
        train_loader_, test_loader_, classes, n_iter = \
                gen_mnist_dataset()
        dataiter = iter(train_loader_)
        print("sample size of the training loader mnist {}".\
                format(len(train_loader_)))
        print("sample size of the testing loader mnist {}".\
                format(len(test_loader_)))
        print("iter per epoch is mnist {}".\
                format(n_iter))
        images, labels = dataiter.next()
        cv_helper.im_show_plt(cv_helper.make_img_grid(images))
        print("len labels {}".format(len(labels)))
        print(' '.join('%5s' % classes[j] for j in range(4)))
        cv_helper.im_toggle_show_plt()

if __name__ == "__main__":
    unittest.main()

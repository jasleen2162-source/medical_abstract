from src.gpu_setup import *

from src.visualize import *

from src.train import *

from src.evaluate import *


def main():

    setup_gpu()

    show_devices()

    show_samples()

    visualize_labels()

    train()

    evaluate_model()


if __name__=="__main__":

    main()
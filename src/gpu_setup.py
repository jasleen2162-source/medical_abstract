import tensorflow as tf


def setup_gpu():

    gpus=tf.config.list_physical_devices(

        "GPU"

    )

    if gpus:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(

                gpu,

                True

            )

        print(

            "GPU Enabled"

        )

    else:

        print(

            "CPU Mode"

        )


def show_devices():

    print(

        tf.config.list_physical_devices()

    )
import numpy as np
import matplotlib.pyplot as plt


def plot_history(histories, figures):
    plt.figure(1, figsize=(4, 6))
    index = 0

    for name, history in histories:
        for keys in figures:
            index = index + 1
            plt.subplot(2, 1, index)
            plt.xlabel('Epochs')
            plt.ylabel(keys[0])
            for key in keys:
                plt.plot(history.epoch, history.history[key], label=key.title())
            plt.legend()

    plt.show()

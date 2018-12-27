import numpy as np
import matplotlib.pyplot as plt


def plot_history(histories):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        for key in history.history:
            plt.plot(history.epoch, history.history[key], label=key.title())

    plt.xlabel('Epochs')
    # plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    plt.show()

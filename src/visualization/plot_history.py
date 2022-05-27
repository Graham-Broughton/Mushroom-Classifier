def plot_history(accuracy_or_loss):
    plt.plot(history.history[accuracy_or_loss], label=accuracy_or_loss)
    plt.plot(history.history['val'+accuracy_or_loss], label='Validation '+accuracy_or_loss)
    plt.xlabel('Epoch')
    plt.ylabel(accuracy_or_loss)
    plt.title(accuracy_or_loss+' and Validation '+accuracy_or_loss)
    plt.legend()
    plt.show()

from math import sqrt

import matplotlib.pyplot as plt
import wandb


def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if (
        numpy_labels.dtype == object
    ):  # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels


def title_from_label_and_target(label, correct_label, class_dict):
    if correct_label is None:
        return class_dict[label], True
    correct = label == correct_label
    return (
        "{} [{}{}{}]".format(
            class_dict[label],
            "OK" if correct else "NO",
            "\u2192" if not correct else "",
            class_dict[correct_label] if not correct else "",
        ),
        correct,
    )


def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis("off")
    plt.imshow(image)
    if len(title) > 0:
        plt.title(
            title,
            fontsize=int(titlesize) if not red else int(titlesize / 1.2),
            color="red" if red else "black",
            fontdict={"verticalalignment": "center"},
            pad=int(titlesize / 1.5),
        )
    return (subplot[0], subplot[1], subplot[2] + 1)


def display_batch_of_images(databatch, class_dict, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(sqrt(len(images)))
    cols = len(images) // rows

    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

    # display
    for i, (image, label) in enumerate(
        zip(images[: rows * cols], labels[: rows * cols])
    ):
        title = "" if label is None else class_dict[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(
                predictions[i], label, class_dict
            )
        dynamic_titlesize = (
            FIGSIZE * SPACING / max(rows, cols) * 40 + 3
        )  # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(
            image, title, subplot, not correct, titlesize=dynamic_titlesize
        )

    # layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()


def display_confusion_matrix(cmat, score, precision, recall, class_dict):
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    ax.matshow(cmat, cmap="Reds")
    ax.set_xticks(range(len(class_dict)))
    ax.set_xticklabels(class_dict, fontdict={"fontsize": 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(class_dict)))
    ax.set_yticklabels(class_dict, fontdict={"fontsize": 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += "f1 = {:.3f} ".format(score)
    if precision is not None:
        titlestring += "\nprecision = {:.3f} ".format(precision)
    if recall is not None:
        titlestring += "\nrecall = {:.3f} ".format(recall)
    if titlestring != "":
        ax.text(
            101,
            1,
            titlestring,
            fontdict={
                "fontsize": 18,
                "horizontalalignment": "right",
                "verticalalignment": "top",
                "color": "#804040",
            },
        )
    plt.show()


def display_training_curves(training, validation, title, subplot, CFG, save_time):
    if subplot % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(10, 10), facecolor="#F0F0F0")
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor("#F8F8F8")
    ax.plot(training)
    ax.plot(validation)
    ax.set_title(f"model {title}")
    ax.set_ylabel(title)
    # ax.set_ylim(0.28,1.05)
    ax.set_xlabel("epoch")
    ax.legend(["train", "valid."])
    wandb.log({"chart": plt})
    path = CFG.ROOT / "images" / CFG.MODEL
    path.mkdir(exist_ok=True)
    plt.savefig(path / f"{title}-{save_time}.png")


# def plot_training(history, fold, CFG):
#     plt.figure(figsize=(15, 5))
#     plt.plot(np.arange(CFG.EPOCHS[fold]), history['auc'], '-o', label='Train AUC', color='#ff7f0e')
#     plt.plot(np.arange(CFG.EPOCHS[fold]), history['val_auc'], '-o', label='Val AUC', color='#1f77b4')
#     x = np.argmax(history.history['val_auc'])
#     y = np.max(history.history['val_auc'])
#     xdist = plt.xlim()[1] - plt.xlim()[0]
#     ydist = plt.ylim()[1] - plt.ylim()[0]
#     plt.scatter(x, y, s=200, color='#1f77b4')
#     plt.text(x - 0.03 * xdist, y - 0.13 * ydist, 'max auc\n%.2f' % y, size=14)
#     plt.ylabel('AUC', size=14)
#     plt.xlabel('Epoch', size=14)
#     plt.legend(loc=2)
#     plt2 = plt.gca().twinx()
#     plt2.plot(np.arange(CFG.EPOCHS[fold]), history['loss'], '-o', label='Train Loss', color='#2ca02c')
#     plt2.plot(np.arange(CFG.EPOCHS[fold]), history['val_loss'], '-o', label='Val Loss', color='#d62728')
#     x = np.argmin(history['val_loss'])
#     y = np.min(history['val_loss'])
#     ydist = plt.ylim()[1] - plt.ylim()[0]
#     plt.scatter(x, y, s=200, color='#d62728')
#     plt.text(x - 0.03 * xdist, y + 0.05 * ydist, 'min loss', size=14)
#     plt.ylabel('Loss', size=14)
#     plt.title(
#         'FOLD %i - Image Size %i, EfficientNet B%i'
#         % (fold + 1, CFG.IMG_SIZES[fold], EFF_NETS[fold]),
#         size=18,
#     )
#     plt.legend(loc=3)
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def acc_mag_plot(magnitudes, accuracies, class_names):
    assert len(accuracies)==len(class_names)
    



if __name__ == '__main__':
    class_names = ['galaxy', 'star', 'qso']
    accuracies = [
        [1.0, 1.0, 0.9887640449438202, 1.0, 1.0, 0.9958847736625515, 0.9966329966329966, 0.9911242603550295, 0.9939577039274925, 0.9895833333333334, 0.986013986013986, 0.9550827423167849, 0.9337606837606838, 0.8907563025210085, 0.8503118503118503, 0.784741144414169, 0.796812749003984, 0.7241379310344828, 0.7621621621621621, 0.6666666666666666, 0.6428571428571429, 0.5714285714285714],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.996268656716418, 1.0, 0.9961832061068703, 0.996415770609319, 0.9601593625498008, 0.8960573476702509, 0.9070796460176991, 0.8272727272727273, 0.8863636363636364, 0.7558139534883721, 0.8247422680412371, 0.7586206896551724, 0.7272727272727273, 0.6875, 0.42857142857142855],
        [1.0, 1.0, 0.0, 0.4, 0.6666666666666666, 0.75, 1.0, 1.0, 0.8, 0.7894736842105263, 0.8709677419354839, 0.6666666666666666, 0.66, 0.647887323943662, 0.5662650602409639, 0.5833333333333334, 0.5982905982905983, 0.47540983606557374, 0.42758620689655175, 0.29411764705882354, 0.43902439024390244, 0.3888888888888889]
    ]
    magnitudes = [16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5]

    acc_mag_plot(magnitudes, accuracies, class_names)
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}


def get_sets(df, target='classes', n_bands=12, filters=None):
    """
    receives:
    * df            pandas dataframe
    * filters       dict with optional min-max values, e.g. {'feature1': [min1, max1], 'feature2': [min2, max2]}
    * obj_list      list of object ids (used to check existing files)
    * mode          'classes' to return class labels or 'magnitudes' to return 12 magnitudes; other string values will return None for y/labels

    returns: (X, y, labels) triplet, where
    * X is a list of object ids
    * y are integer-valued labels
    * labels is a dict mapping each id to its label, e.g. {'x1': 0, 'x2': 1, ...}
    """
    # print('original set size', df.shape)
    if filters is not None:
        for key, val in filters.items():
            df = df[df[key].between(val[0], val[1])]
            # print('set size after filters', df.shape)
    X = df.id.values
    
    if target=='classes':
        y = df['class'].apply(lambda c: class_map[c]).values
        y = to_categorical(y, num_classes=3)
    elif target=='magnitudes':
        if n_bands==5:
            y = df[['u','g','r','i','z']].values
        else:
            y = df[['u','f378','f395','f410','f430','g','f515','r','f660','i','f861','z']].values
        # y = y/30
    elif target=='redshifts':
        y = df[['redshift_base', 'redshift_exp']].values
    else:
        return X, _, _

    labels = dict(zip(X, y))
    return X, y, labels


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
    # classes = classes[unique_labels(y_true, y_pred)]
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


def plot_accuracy_mags(magnitudes, accuracies, class_names, title=None):
    assert len(accuracies)==len(class_names)
    fig, ax = plt.subplots()
    for ix in range(len(accuracies)):
        scatter = ax.scatter(magnitudes, accuracies[ix], label=class_names[ix], alpha=0.7)
    ax.legend(loc='lower right')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('magnitude')
    ax.set_title(title)
    ax.grid(True)
    plt.show()


def plot_loss_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    class_names = ['galaxy', 'star', 'qso']

    # accuracies = [
    #     [1.0, 1.0, 0.9887640449438202, 1.0, 1.0, 0.9958847736625515, 0.9966329966329966, 0.9911242603550295, 0.9939577039274925, 0.9895833333333334, 0.986013986013986, 0.9550827423167849, 0.9337606837606838, 0.8907563025210085, 0.8503118503118503, 0.784741144414169, 0.796812749003984, 0.7241379310344828, 0.7621621621621621, 0.6666666666666666, 0.6428571428571429, 0.5714285714285714],
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.996268656716418, 1.0, 0.9961832061068703, 0.996415770609319, 0.9601593625498008, 0.8960573476702509, 0.9070796460176991, 0.8272727272727273, 0.8863636363636364, 0.7558139534883721, 0.8247422680412371, 0.7586206896551724, 0.7272727272727273, 0.6875, 0.42857142857142855],
    #     [1.0, 1.0, 0.0, 0.4, 0.6666666666666666, 0.75, 1.0, 1.0, 0.8, 0.7894736842105263, 0.8709677419354839, 0.6666666666666666, 0.66, 0.647887323943662, 0.5662650602409639, 0.5833333333333334, 0.5982905982905983, 0.47540983606557374, 0.42758620689655175, 0.29411764705882354, 0.43902439024390244, 0.3888888888888889]
    # ]
    # magnitudes = [16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5]

    # plot_accuracy_mags(magnitudes, accuracies, class_names, 'image classifier trained with magnitudes in [16,19]')

    # accuracies = [
    #     [0.9230769230769231, 0.9620253164556962, 0.9550561797752809, 0.9491525423728814, 0.9793103448275862, 0.9300411522633745, 0.9595959595959596, 0.9260355029585798, 0.9667673716012085, 0.9479166666666666, 0.9300699300699301, 0.9148936170212766, 0.9166666666666666, 0.9411764705882353, 0.9313929313929314, 0.9100817438692098, 0.8884462151394422, 0.9233716475095786, 0.8972972972972973, 0.8095238095238095, 0.8571428571428571, 1.0],
    #     [0.9788732394366197, 0.9931972789115646, 0.9496402877697842, 0.9597701149425287, 0.9736842105263158, 0.9545454545454546, 0.947136563876652, 0.9626865671641791, 0.952755905511811, 0.916030534351145, 0.8781362007168458, 0.8884462151394422, 0.8315412186379928, 0.7522123893805309, 0.740909090909091, 0.7329545454545454, 0.7267441860465116, 0.8041237113402062, 0.6781609195402298, 0.7272727272727273, 0.3125, 0.2857142857142857],
    #     [0.0, 0.5, 0.0, 0.2, 0.3333333333333333, 0.5, 0.6666666666666666, 0.625, 0.5, 0.5263157894736842, 0.6451612903225806, 0.5952380952380952, 0.6, 0.7464788732394366, 0.6144578313253012, 0.7037037037037037, 0.7435897435897436, 0.7213114754098361, 0.7586206896551724, 0.8823529411764706, 0.926829268292683, 0.8888888888888888]

    # ]
    # magnitudes = [16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5]

    # plot_accuracy_mags(magnitudes, accuracies, class_names, 'catalog classifier trained with magnitudes in [16,19]')

    # accuracies = [
    #     [0.9230769230769231, 0.9113924050632911, 0.9550561797752809, 0.9576271186440678, 0.9724137931034482, 0.9423868312757202, 0.9595959595959596, 0.9230769230769231, 0.9335347432024169, 0.9296875, 0.9114219114219114, 0.8983451536643026, 0.9038461538461539, 0.9285714285714286, 0.9106029106029107, 0.9073569482288828, 0.896414342629482, 0.9386973180076629, 0.9135135135135135, 0.8690476190476191, 0.9761904761904762, 1.0],
    #     [0.9225352112676056, 0.9115646258503401, 0.841726618705036, 0.9195402298850575, 0.9263157894736842, 0.898989898989899, 0.8854625550660793, 0.8805970149253731, 0.8622047244094488, 0.8282442748091603, 0.7706093189964157, 0.7848605577689243, 0.7383512544802867, 0.6769911504424779, 0.6545454545454545, 0.6590909090909091, 0.6453488372093024, 0.7731958762886598, 0.6206896551724138, 0.5681818181818182, 0.375, 0.2857142857142857],
    #     [0.0, 0.5, 0.0, 0.0, 0.0, 0.25, 0.6666666666666666, 0.5, 0.4, 0.47368421052631576, 0.5483870967741935, 0.5952380952380952, 0.66, 0.7323943661971831, 0.5903614457831325, 0.6388888888888888, 0.7094017094017094, 0.5819672131147541, 0.6689655172413793, 0.7352941176470589, 0.7317073170731707, 0.7777777777777778]
    # ]
    # magnitudes = [16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5]

    # plot_accuracy_mags(magnitudes, accuracies, class_names, 'catalog classifier trained with all magnitudes')

    # accuracies = [
    #     [1.0, 1.0, 0.9775280898876404, 0.9915254237288136, 0.993103448275862, 0.9917695473251029, 0.9966329966329966, 0.9911242603550295, 0.9939577039274925, 0.9895833333333334, 0.9906759906759907, 0.9858156028368794, 0.9743589743589743, 0.9642857142857143, 0.9708939708939709, 0.9400544959128065, 0.9282868525896414, 0.9540229885057471, 0.9513513513513514, 0.8690476190476191, 0.8571428571428571, 0.8571428571428571],
    #     [0.9929577464788732, 1.0, 0.9928057553956835, 1.0, 1.0, 1.0, 0.9911894273127754, 0.996268656716418, 0.984251968503937, 0.9847328244274809, 0.989247311827957, 0.9800796812749004, 0.953405017921147, 0.9424778761061947, 0.9090909090909091, 0.9488636363636364, 0.8546511627906976, 0.9690721649484536, 0.8850574712643678, 0.8409090909090909, 0.5, 0.7142857142857143],
    #     [0.0, 1.0, 0.0, 0.4, 0.6666666666666666, 0.75, 1.0, 0.625, 0.9, 0.7368421052631579, 0.8064516129032258, 0.7380952380952381, 0.7, 0.8028169014084507, 0.7108433734939759, 0.6851851851851852, 0.7692307692307693, 0.6147540983606558, 0.6551724137931034, 0.5980392156862745, 0.6829268292682927, 0.5]
    # ]
    # magnitudes = [16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5]

    # plot_accuracy_mags(magnitudes, accuracies, class_names, 'image classifier trained with all magnitudes')

    accuracies = [
        [0.9423076923076923, 0.9240506329113924, 0.9662921348314607, 0.940677966101695, 0.9172413793103448, 0.9300411522633745, 0.9292929292929293, 0.9142011834319527, 0.8549848942598187, 0.8619791666666666, 0.7855477855477856, 0.7659574468085106, 0.7649572649572649, 0.6953781512605042, 0.6112266112266113, 0.47956403269754766, 0.3107569721115538, 0.21455938697318008, 0.12972972972972974, 0.13095238095238096, 0.09523809523809523, 0.14285714285714285],
        [0.9929577464788732, 0.9931972789115646, 0.9928057553956835, 0.9885057471264368, 1.0, 1.0, 0.9955947136563876, 0.9813432835820896, 0.9921259842519685, 0.9770992366412213, 0.978494623655914, 0.9442231075697212, 0.910394265232975, 0.8495575221238938, 0.7181818181818181, 0.6193181818181818, 0.5930232558139535, 0.6391752577319587, 0.4827586206896552, 0.29545454545454547, 0.125, 0.14285714285714285],
        [0.0, 0.0, 1.0, 0.4, 0.3333333333333333, 0.5, 1.0, 0.5, 0.6, 0.6842105263157895, 0.7741935483870968, 0.7619047619047619, 0.7, 0.8450704225352113, 0.9036144578313253, 0.9444444444444444, 0.9316239316239316, 0.9836065573770492, 0.9862068965517241, 0.9901960784313726, 1.0, 1.0]
    ]
    magnitudes = [16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5]

    plot_accuracy_mags(magnitudes, accuracies, class_names, 'image classifier trained with class weights [1, 1.25, 5]')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import re
import numpy as np

def plot_distribution(data, caption) :

    fig = plt.figure(figsize=(15,8))
    plt.rc('font', size=30) 
    sns.countplot(x = data.labels)
    plt.title(caption + " data")
    plt.ylabel('Frequency')
    plt.xlabel('Labels')
    plt.savefig('figures/distribution_'+caption+'.pdf', dpi='figure')
    plt.show()


def plot_length_sentences(data, caption) :

    fig = plt.figure(figsize=(12,8))
    plt.rc('font', size=30) 
    sns.distplot(x = [len(re.findall('[a-z-]+', text, flags=re.I)) for text in data.text], 
                 kde=False, rug=False)
    plt.title(caption + " data")
    plt.ylabel('Frequency')
    plt.xlabel('Number of words')
    plt.tight_layout()
    plt.savefig('figures/len_sentences_'+caption+'.pdf', dpi='figure')
    plt.show()


def plot_accuracy_curve(History, network) :    
    fig = plt.figure(figsize=(10,8))
    plt.rc('font', size=15)
    plt.plot(History.history['accuracy'], label='accuracy', marker='o')
    plt.plot(History.history['val_accuracy'], label='val_accuracy', marker='o')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/accuracy'+network+'.pdf', dpi='figure')
    plt.show()


def plot_loss_curve(History, network) :

    fig = plt.figure(figsize=(10,8))
    plt.rc('font', size=15)
    plt.plot(History.history['loss'], label='loss', marker='o')
    plt.plot(History.history['val_loss'], label='val_loss', marker='o')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('figures/loss_'+network+'.pdf', dpi='figure')
    plt.legend()
    plt.show()


def confusion_matrix_NN(model, val_data, X_val_padded, labels_dict, network, multi=True):

    fig = plt.figure(figsize=(20,20))
    plt.rc('font', size=15)
    if(multi):
      cf = confusion_matrix(val_data.labels, [np.argmax(x) for x in model.predict(X_val_padded)])
    else:
      cf = confusion_matrix(val_data.labels, [round(x[0]) for x in model.predict(X_val_padded)])
    ConfusionMatrixDisplay(confusion_matrix=cf).plot()
    plt.xticks(range(len(labels_dict.keys())),labels_dict, fontsize=12)
    plt.yticks(rotation = 45)
    plt.yticks(range(len(labels_dict.keys())), labels_dict, fontsize=12)
    plt.savefig('figures/confusion_matrix'+network+'.pdf', dpi='figure')
    plt.show()

import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def comp_prior(dataset, Y):

    '''
    Computing the prior
    '''

    classes = sorted(list(dataset[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(dataset[dataset[Y]==i])/len(dataset))
    return prior

def comp_likelihood(dataset, feat_name, feat_val, Y, label, types):

    '''
    Computing likelihood depending on the type of variable requested
    '''
    p_x_given_y = 0

    dataset = dataset[dataset[Y]==label]

    if types == 'categorical':
        p_x_given_y = len(dataset[dataset[feat_name]==feat_val]) / len(dataset)
    
    elif types == 'gaussian':
        mean, std = dataset[feat_name].mean(), dataset[feat_name].std()
        p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))

    return p_x_given_y  

def naive_bayes(train_dataset, X_Test, Y, types):

    '''
    get feature names
    calculate prior
    loop over every data sample
    calculate likelihood
    calculate posterior probability (numerator only)
    '''
    
    features = list(train_dataset.columns)[:-1]

    prior = comp_prior(train_dataset, Y)

    y_pred = []
    for x in X_Test:

        classes = sorted(list(train_dataset[Y].unique()))
        likelihood = [1]*len(classes)
        for c in range(len(classes)):
            for i in range(len(features)):
                likelihood[c] *= comp_likelihood(train_dataset, features[i], x[i], Y, classes[c], types)

        post_prob = [1]*len(classes)
        for c in range(len(classes)):
            post_prob[c] = likelihood[c] * prior[c]

        y_pred.append(np.argmax(post_prob))

    return np.array(y_pred)

def CV (dataset, target, folds, t):

    accuracy = []
    precision = []
    recall = []
    f1 = []
                
    for i in range(folds):
        train, test = train_test_split(dataset, test_size = 1/folds, random_state = i)

        X_test = test.iloc[:,:-1].values
        y_test = test.iloc[:,-1].values
        y_pred = naive_bayes(train, X_Test=X_test, Y=target, types=t)

        accuracy.append(accuracy_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred, average='weighted'))
        f1.append(f1_score(y_test, y_pred, average='weighted'))
        precision.append(precision_score(y_test, y_pred, average='weighted'))
    
    return np.mean(accuracy), np.mean(recall), np.mean(f1), np.mean(precision)

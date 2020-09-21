import tensorflow as tf
from tensorflow.keras import metrics
from keras import backend as K
import numpy as np


def top_1_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=1)

def top_2_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=2)

def top_10_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=10)

def top_50_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=50)


def meaure_topk_for_regression(y_true,y_pred,k):
    'Measure top 10 accuracy for regression'
    c = 0
    for i in range(len(y_pred)):
        # shape of each elemnt is (256,)
        A = y_true[i]
        B = y_pred[i]
        top_predictions = B.argsort()[-10:][::-1]
        best = np.argmax(A)
        if best in top_predictions:
             c +=1

    return c/len(y_pred)

def R2_metric(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



def seperate_metric_in_out_train(model,x_train,y_train_true,x_test, y_test_true):
    """There are some classes which are not included in the validation set.
    This function evalutes the performance of in-train and not-in-train classes
    + It shows the apperance of diffrent classes in our predictions"""

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    k =1     # We consider the best beam
    labels_in_train = [i.argsort()[-k:][::-1][0] for i in y_train_true]
    labels_in_test = [i.argsort()[-k:][::-1][0] for i in y_test_true]

    unquie_labels_in_train = list(set(labels_in_train))
    unquie_labels_in_test = list(set(labels_in_test))
    # print('emerged labels in train set are',labels_in_train,'emerged labels in test set are',labels_in_test)


    not_in_train = [c for c in unquie_labels_in_test if c not in unquie_labels_in_train]
    print('These labels are not emerged in the test (validation) set',not_in_train)


    in_train = in_train_correct = in_train_wrong = 0
    not_in_train = not_in_train_correct = not_in_train_wrong = 0

    for count in range(len(y_test_pred)):
        prediction = y_test_pred[count].argsort()[-k:][::-1]
        true = y_test_true[count].argsort()[-k:][::-1]

        if true in labels_in_train:
            in_train +=1
            if true == prediction:
                in_train_correct+=1
            else:
                in_train_wrong+=1

        elif true not in labels_in_train:
            not_in_train+=1
            if true == prediction:
                not_in_train_correct+=1
            else:
                not_in_train_wrong+=1
    print('{} samples of test set are in training set, {} correctly predicted,{} wrongly predicted'.format(in_train,in_train_correct,in_train_wrong))
    print('{} samples of test set are not in training set,{} correctly predicted,{} wrongly predicted'.format(not_in_train,not_in_train_correct,not_in_train_wrong))

    print('*************Count apperance of classes in our prediction*************')
    max_true = y_test_true.argmax(axis=1)
    max_pred = y_test_pred.argmax(axis=1)

    print(max_true.shape,max_pred.shape)
    Occurrence_true = {i:len(np.where(max_true==i)[0]) for i in np.unique(max_true)}
    Occurrence_pred = {i:len(np.where(max_pred==i)[0]) for i in np.unique(max_pred)}
    print('Occurrence_true_labels',Occurrence_true)
    print('Occurrence_pred_labels',Occurrence_pred)










#!/usr/bin/env pyth on3 Spyder Editor
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:06:22 2018

@author : Gaurav Gahukar
        : caffeine110
        
AIM     : To build Ann which predicts which Customer is most likely to leave the job
        : implimenting ANN in Sklearn

"""

###############################################################################
### importing data
def get_Data():
    ### importing data
    from preprocessing import X_train, X_test, Y_train, Y_test
    return X_train, X_test, Y_train, Y_test



###############################################################################
### Building model
def build_Model():
    ### Importing keras libraries and packages
    #import keras
    from keras.models import Sequential
    from keras.layers import Dense


    #initialisation of ANN model
    clf_model = Sequential()
    
    #Adding input layer and one Hiden layer 
    clf_model.add(Dense(output_dim = 11, init = 'uniform', activation= 'relu', input_dim = 11))

    #Adding the second hidden layer
    clf_model.add(Dense(output_dim = 8, init = 'uniform', activation='relu'))
    
    #Adding the second hidden layer
    clf_model.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))
    
    #Adding the output layer
    clf_model.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
    clf_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    return clf_model



###############################################################################
def load_Weights(saved_model):
    saved_model.load_weights('checkpoints/my_clf_Model_weights.h5')
    return saved_model


###############################################################################
def display_CM(Y_test,Y_pred):
    #making the confusion Metrix
    from sklearn.metrics import confusion_matrix
    
    #creating cofusion metrix
    cm = confusion_matrix(Y_test,Y_pred)

    print('\n')
    print('Confusion Matrix Obtained is : ')
    print(cm)



###############################################################################
def get_Predictions(saved_model, X_test):    
    Y_pred = saved_model.predict(X_test)
    Y_pred = (Y_pred > 0.5)
    return Y_pred




###############################################################################
### Accuracy graph
def plot_Accuracy_Graph(Y_test, Y_pred):
    y_original = Y_test[50:100]
    y_predicted = Y_pred[50:100]
    
    ### importing matplotlib
    import matplotlib.pyplot as plt

    plt.plot(y_original, 'r')
    plt.plot(y_predicted, 'b')
    plt.ylabel('predicted-b/original-r')
    plt.xlabel('n')
    plt.legend(['predicted', 'original'], loc='upper left')

    plt.show()



###############################################################################
### Accure Scores
def accuracy_Score(Y_test, Y_pred):
    import sklearn.metrics
    ### Calculating the Varience Score
    res1 = sklearn.metrics.explained_variance_score(Y_test, Y_pred)
    print("Varience Score is : ",res1)





###############################################################################
def main():
    ### function call to get data
    X_train, X_test, Y_train, Y_test = get_Data()
    
    ### function call to build MOdel
    saved_model = build_Model()
    
    
    saved_model = load_Weights(saved_model)
    
    Y_pred = get_Predictions(saved_model, X_test)
    display_CM(Y_test,Y_pred)
    pass






###############################################################################
if __name__ == '__main__':
    main()


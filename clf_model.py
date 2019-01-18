#!/usr/bin/env pyth on3 Spyder Editor
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:06:22 2018

@author : Gaurav Gahukar

AIM     : TO build Ann which predicts which Customer is most likely to leave the job
        : implimenting ANN in Sklearn

"""


### importing data
def get_Data():
    ### importing data
    from preprocessing import X_train, X_test, Y_train, Y_test
    return X_train, X_test, Y_train, Y_test



### Building model
def build_Model():
    ### Importing keras libraries and packages
    #import keras
    from keras.models import Sequential
    from keras.layers import Dense


    #initialisation of ANN model
    clf_model = Sequential()
    
    #Adding input layer and one Hiden layer 
    clf_model.add(Dense(output_dim = 8, init = 'uniform', activation= 'relu', input_dim = 11))
    
    #Adding the second hidden layer
    clf_model.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))
    
    #Adding the output layer
    clf_model.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
    clf_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    return clf_model


def save_Model(clf_model):
    #from keras.models import load_model
    clf_model.save('saved_model/my_clf_Model.h5')  # creates a HDF5 file 'my_model.h5'


def save_Weights(clf_model):
    clf_model.save_weights('checkpoints/my_clf_Model_weights.h5')


# summarize history for loss
def plot_Loss(history):
    ### importing matplotlib
    import matplotlib.pyplot as plt
        
    #dict_keys(['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

# summarize history for accuracy
def plot_Accuracy(history):
    ### importing matplotlib
    import matplotlib.pyplot as plt
    
    #dict_keys(['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error'])
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('validation accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


### main method
def main():
    ### function call to get data    
    X_train, X_test, Y_train, Y_test = get_Data()
    
    
    ### function call to build MOdel
    clf_model = build_Model()


    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)


    ### fitting the model
    history = clf_model.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100, validation_split=0.2, callbacks=[early_stopping])

    save_Model(clf_model)
    save_Weights(clf_model)

    plot_Loss(history)
    plot_Accuracy(history)

    ### printing the history
    print(history.history)

    





if __name__ == "__main__":
    main()

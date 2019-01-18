
## AIM	: Predictive analysis of Customer behaviour using Machine Learning.
	: ( Which Customer is most likely to leave the company or Stays...)


### Author	: caffeine110

### Introduction
It very Hard to predict the behaviour of customer for machine
problem can be solved using the Neural Network


### Keywords 
Keywords : Machine Learning, Data Analysis, Satastics, DNN, Numpy, Pandas.

## Tools
PreRequirements :
		 LIBRARIES	: Pandas,Numpy, Sklearn, Keras, TensorFlow, matplotlib, csv.
		 IDE		: spyder

###
Abstraact	: We have studied the data manipulation libraries such as Numpy and Pandas for handling the huge dataset of Customers.
		  using matplot library we can visualise all the implimented modules.
		  Using sk-learn we can import test-train split method which divides the whole data into test and train cases.
		  Using keras we can build the DNN model with Sequential layers.
		  TensorFlow is an alternative library which allows to create ML model using Estemators and Tensors.


# procedure to run
Procedure : 

	1). Exctraction :
		Dataset is exctrated from kaggle
	2). Preporcessing
		Run the preprocessing.py file to preprocess the downloaded data.
	3). Model Training
		Run the clf_model.py file to fit data to model.
		While the model is trained program is under exicution and after complition apply the prediction steps.
	4). prediction
		To predict the behaviour of cusotmer put the data tupule in X_test cases.
		Output is shown in single Binary Value (1 / 0 ) and ( yes / no ).


# Evaluation Plan

As this is a Logistic Regression problem it is difficult to measure performance of Regressor  model than the Classification One.
So we have mesure the performance of model using Confusion Metrix.


### key Metrics :

Epoch 1/200
6000/6000 [==============================] - 2s 355us/step - loss: 0.4748 - acc: 0.7958 - val_loss: 0.4321 - val_acc: 0.7967

Epoch 32/200
6000/6000 [==============================] - 1s 220us/step - loss: 0.3969 - acc: 0.8367 - val_loss: 0.4055 - val_acc: 0.8287

Loss decreased from 0.4748 to 0.3969
Validation Loss decreased from 0.4321 to 0.4055

Accurcy increased from 0.7958 to 0.8367
Validation Accuracy increased from 0.7967 to 0.8287


Variance—
	In terms of linear regression, variance Is a measure of how far observed
	values differ from the average of predicted values.
	Idely it is 1

Mean Square Error (MSE)—
	It is the average of the square of the errors.
	The larger the number the larger the error.

Absolute errorse(AE)—
	It is a difference between two continues variables



# Optimisation :

### Parameter Tuning

We have tued the Parameters from the 

	Train-Test split from 60-40 ... 80-20 and get the best accuracy at 75-25

	Varing from 1... we choose Dense Layers : 3

	Number of Neurons in Layer each layers :
	We got best accuracy at layers at :
		: 11 Neurons at input layer as Parameters are 11
		: 8 Neurons at First Hidden layer
		: 6 Neurons at Second Hidden layer
		: 1 Neuron at output layer as there is only 1 Output Price

	Tuned No of epoches 0 to 100 applied the Early_Stopping to stop model Training at paitence 4
	Batch size 10

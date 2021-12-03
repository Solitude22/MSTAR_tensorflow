"""
Trimmed version of mstar_network.py from the original repo.
https://github.com/hamza-latif/MSTAR_tensorflow

mstarnet has been replaced with a deeper CNN with 5 convolution layers. This model has been used 
in the past on other projects and has proven to be a high performing model without high complexity.

Alan Kittel
The Aerospace Corporation
12/3/21
"""

import tensorflow as tf
import numpy as np
from data import DataHandler
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

"""
Implementation of mstartnet using Tensorflow.
mstarnet is a basic conv-net in mstar_network.py. Here a similarly structured but deeper network is used.
msarnet is a sequential convolutional network (CNN) with 5 convolutional layers and a dropout layer.

Input:
	classes -> integer for the number of classes in the dataset
Output:
	constructed TensorFlow model ready for training
"""
def mstarnet(classes):
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1), padding='same'))
	model.add(layers.MaxPooling2D())
	model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
	model.add(layers.MaxPooling2D())
	model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
	model.add(layers.MaxPooling2D())
	model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
	model.add(layers.MaxPooling2D())
	model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
	model.add(layers.MaxPooling2D())

	model.add(layers.Dropout(0.2))
	model.add(layers.Flatten())
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.Dense(classes))
	
	return model

"""
Trains a CNN model to classify images from the MSTAR dataset.

Input:
	data_handler -> DataHandler object containing the MSTAR data
	modelSave -> filepath for saving the TensorFlow model
	targets -> string array of target names in alphabetical order
	num_epochs -> number of training epochs, default is 50
Output:
	none explicitly, but TensorFlow model is saved to specified location
	and accuracy, loss, and confusion matrix plots are displayed.
	At user discretion, plots may be manually saved from the pop-up windows.
"""
def train_nn_tflearn(data_handler,modelSave,targets,num_epochs=50):

	classes = data_handler.num_labels

	# call data_handler function to get training data and reshape it
	X, Y = data_handler.get_all_train_data()
	X = X[:,:128*128]
	X = X.reshape([-1,128,128,1])
	Y = np.reshape(Y, (-1,1))

	# call data_handler function to get testing data and reshape it
	X_test, Y_test = data_handler.get_test_data()
	X_test = X_test[:,:128*128]
	X_test = X_test.reshape([-1,128,128,1])
	# data handler's get_test_data returns the label array in a different format than
	# get_all_train_data, these next few lines are for reshaping it
	Y_test_new = np.zeros((Y_test.shape[0], 1))
	for dim in range(Y_test.shape[1]):
		Y_test_new[Y_test[:,dim]==1,:] = dim
	Y_test = Y_test_new

	"""
	Below is new code for Tensorflow model
	"""
	# create network
	model = mstarnet(classes)

	# create training model
	model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

	# set checkpoint to save the best model
	checkpoint = tf.keras.callbacks.ModelCheckpoint(modelSave, save_best_only=True, mode='max', verbose=1, monitor='val_acc')

	# see architecture summary of the model
	model.summary()

	# train model
	history = model.fit(X, Y, epochs=num_epochs, validation_data=(X_test, Y_test), callbacks=[checkpoint])

	# evaluate
	model.evaluate(X_test, Y_test, verbose=2)

	"""Display the visualization of the training accuracy/loss, validation accuracy/loss and confusion matrix."""
	# summarize history for accuracy
	plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='lower right')

	# summarize history for loss
	plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper right')

	# compute confusion matrix
	plt.figure()
	model = load_model(modelSave)
	y_pred = np.argmax(model.predict(X_test), axis=-1)
	cf_matrix = confusion_matrix(Y_test, y_pred, normalize='true')

	# display confusion matrix nicely formatted as a heatmap with the seaborn library
	ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='.5g')
	ax.set_title('Confusion Matrix\n\n')
	ax.set_xlabel('\nPredicted Values')
	ax.set_ylabel('Actual Values ')
	ax.xaxis.set_ticklabels(targets)
	ax.yaxis.set_ticklabels(targets)

	# show all the plots
	plt.show()

"""
Make modifcations here.
"""
if __name__ == '__main__':
	""" Change to point to the output folder location produced by readmstar.py """
	# folder location of data produced by readmstar.py
	data_folder = "D:\My Documents\Work\Aerospace\MSTAR_tensorflow\output"

	# number of batch files in output/ folder. Do not need to change from 1.
	num_batch_files = 1

	# batch size for model training
	mini_batch_size = 32

	# number of epochs for model training
	epochs = 20

	""" Change to desired save location """
	# file to save trained model to
	modelSave = "models/mstar_targets/mstarnet/temp.h5"

	""" Change for mixed targets vs public targets dataset """
	# string labels for the targets for creating the confusion matrix.
	# Needs to match the order used by readmstar (alphabetical), or the confusion matrix will mislabel rows/columns.
	# targets = ['2S1', 'BRDM2', 'BTR60', 'D7', 'T62', 'ZIL131', 'ZSU23/4'] # for mixed targets dataset
	targets = ['BMP2', 'BTR70', 'T72'] # for public targets dataset

	# call DataHandler to read the data
	handler = DataHandler(data_folder,num_batch_files,mini_batch_size)

	# train the model
	train_nn_tflearn(handler, modelSave, targets, epochs)

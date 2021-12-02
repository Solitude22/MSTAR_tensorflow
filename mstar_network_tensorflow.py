"""
Copy of mstart_network.py using minimal effort to train a Tensorflow model over tflearn.
"""
import tensorflow as tf
import tflearn
import numpy as np
from data import DataHandler
#from network_defs import *
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def example_net(x, classes):
	network = tflearn.conv_2d(x, 32, 3, activation='relu')
	network = tflearn.max_pool_2d(network, 2)
	network = tflearn.conv_2d(network, 64, 3, activation='relu')
	network = tflearn.conv_2d(network, 64, 3, activation='relu')
	network = tflearn.max_pool_2d(network, 2)
	network = tflearn.fully_connected(network, 512, activation='relu')
	network = tflearn.dropout(network, 0.5)
	network = tflearn.fully_connected(network, classes, activation='softmax')

	return network


def trythisnet(x, classes):
	network = tflearn.conv_2d(x,64,5,activation='relu')
	network = tflearn.max_pool_2d(network,3,2)
	network = tflearn.local_response_normalization(network,4,alpha=0.001/9.0)
	network = tflearn.conv_2d(network,64,5,activation='relu')
	network = tflearn.local_response_normalization(network,4,alpha=0.001/9.0)
	network = tflearn.max_pool_2d(network,3,2)
	network = tflearn.fully_connected(network,384,activation='relu',weight_decay=0.004)
	network = tflearn.fully_connected(network,192,activation='relu',weight_decay=0.004)
	network = tflearn.fully_connected(network,classes,activation='softmax',weight_decay=0.0)

	return network

"""
Implementation of mstartnet using Tensorflow.

mstarnet is a basic conv-net in mstar_network.py. Here a similarly structured but deeper network is used.

"""
def mstarnet(x, classes):
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

def resnet1(x, classes, n = 5):
	net = tflearn.conv_2d(x, 16, 3, regularizer='L2', weight_decay=0.0001)
	net = tflearn.residual_block(net, n, 16)
	net = tflearn.residual_block(net, 1, 32, downsample=True)
	net = tflearn.residual_block(net, n - 1, 32)
	net = tflearn.residual_block(net, 1, 64, downsample=True)
	net = tflearn.residual_block(net, n - 1, 64)
	net = tflearn.batch_normalization(net)
	net = tflearn.activation(net, 'relu')
	net = tflearn.global_avg_pool(net)
	# Regression
	net = tflearn.fully_connected(net, classes, activation='softmax')

	return net

def train_nn_tflearn(data_handler,modelSave,targets,num_epochs=50):

	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	#tflearn.init_graph(gpu_memory_fraction=0.5)

	batch_size = data_handler.mini_batch_size
	classes = data_handler.num_labels

	img_prep = tflearn.ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	img_aug = tflearn.ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25)
	#img_aug.add_random_crop([32,32], padding=4)

	x = tflearn.input_data(shape=[None, 128, 128, 1], dtype='float', data_preprocessing=img_prep,
						   data_augmentation=img_aug)
	# x = tf.placeholder('float', [None, 32, 32, 3])
	#y = tf.placeholder('float', [None, 10])

	# test_data, test_labels = data_handler.get_test_data()
	# test_data = test_data.reshape([-1,32,32,3])

	ntrain = data_handler.train_size
	ntest = data_handler.meta['num_cases_per_batch']

	# from tflearn.datasets import cifar10
	# (X, Y), (X_test, Y_test) = cifar10.load_data(dirname="/home/hamza/meh/bk_fedora24/Documents/tflearn_example/cifar-10-batches-py")
	# X, Y = tflearn.data_utils.shuffle(X, Y)
	# Y = tflearn.data_utils.to_categorical(Y, 10)
	# Y_test = tflearn.data_utils.to_categorical(Y_test, 10)

	X, Y = data_handler.get_all_train_data()

	X, Y = tflearn.data_utils.shuffle(X, Y)

	#X = np.dstack((X[:, :128*128], X[:, 128*128:]))
	X = X[:,:128*128]

	#X = X/255.0

	#X = X.reshape([-1,128,128,2])
	X = X.reshape([-1,128,128,1])
	
	Y = tflearn.data_utils.to_categorical(Y,classes)

	X_test, Y_test = data_handler.get_test_data()

	#X_test = np.dstack((X_test[:, :128*128], X_test[:, 128*128:]))
	X_test = X_test[:,:128*128]
	#X_test = X_test/255.0

	#X_test = X_test.reshape([-1,128,128,2])
	X_test = X_test.reshape([-1,128,128,1])
	#network = tflearn.regression(net3(x),optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)
	#mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
	#network = tflearn.regression(resnet1(x),optimizer='sgd',loss='categorical_crossentropy')
	#network = tflearn.regression(resnet1(x,classes),optimizer='adam',loss='categorical_crossentropy')
	# network = tflearn.regression(mstarnet(x,classes),optimizer='adam',loss='categorical_crossentropy')
	#network = tflearn.regression(trythisnet(x,classes),optimizer='adam',loss='categorical_crossentropy')

	"""
	Tensorflow expects a different labeling format
	Tensorflow: (X,1), 2nd dimension is the label starting with 0
	tfleanr: (X,num_classes), 2nd dimension is array of zeros with 1 at the index of the assigned label
	"""
	Y_new = np.zeros((Y.shape[0], 1))
	for dim in range(Y.shape[1]):
		Y_new[Y[:,dim]==1,:] = dim
	Y = Y_new
	Y_test_new = np.zeros((Y_test.shape[0], 1))
	for dim in range(Y_test.shape[1]):
		Y_test_new[Y_test[:,dim]==1,:] = dim
	Y_test = Y_test_new

	"""
	Below is new code for Tensorflow model
	"""
	# create network
	model = mstarnet(x, classes)

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

	# confusion matrix
	plt.figure()
	model = load_model(modelSave)
	y_pred = np.argmax(model.predict(X_test), axis=-1)
	cf_matrix = confusion_matrix(Y_test, y_pred, normalize='true')

	ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='.5g')

	ax.set_title('Confusion Matrix\n\n')
	ax.set_xlabel('\nPredicted Values')
	ax.set_ylabel('Actual Values ')

	ax.xaxis.set_ticklabels(targets)
	ax.yaxis.set_ticklabels(targets)
	plt.show()

if __name__ == '__main__':
	bl = "D:\My Documents\Work\Aerospace\MSTAR_tensorflow\output"
	nb = 1
	mbs = 32
	nep = 20
	modelSave = "models/mstar_targets/mstarnet/mstar_targets_model.h5"
	# targets = ['2S1', 'BRDM2', 'BTR60', 'D7', 'T62', 'ZIL131', 'ZSU23/4']
	targets = ['BMP2', 'BTR70', 'T72']

	handler = DataHandler(bl,nb,mbs)
	train_nn_tflearn(handler,modelSave,targets,nep)

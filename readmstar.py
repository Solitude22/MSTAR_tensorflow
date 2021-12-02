import numpy as np
import pickle
import sys
import os
from fnmatch import fnmatch
import random
from sklearn.model_selection import train_test_split

def readMSTARFile(filename):
# raw_input('Enter the mstar file to read: ')

	#print filename

	f = open(filename, 'rb')
	a = ''

	phoenix_header = []

	while 'PhoenixHeaderVer' not in a:
		a = f.readline().decode('utf-8')

	a = f.readline().decode('utf-8')

	while 'EndofPhoenixHeader' not in a:
		phoenix_header.append(a)
		a = f.readline().decode('utf-8')

	data = np.fromfile(f, dtype='>f4')

	# print data.shape

	# magdata = data[:128*128]
	# phasedata = data[128*128:]

	# if you want to print an image
	# imdata = magdata*255

	# imdata = imdata.astype('uint8')

	targetSerNum = '-'

	for line in phoenix_header:
		#print line
		if 'TargetType' in line:
			targetType = line.strip().split('=')[1].strip()
		elif 'TargetSerNum' in line:
			targetSerNum = line.strip().split('=')[1].strip()
		elif 'NumberOfColumns' in line:
			cols = int(line.strip().split('=')[1].strip())
		elif 'NumberOfRows' in line:
			rows = int(line.strip().split('=')[1].strip())
		
	label = targetType# + '_' + targetSerNum

	roffset = (rows-128)//2
	coffset = (cols-128)//2
	data = data[:rows*cols]
	data = data.reshape((rows,cols))
	data = data[roffset:(128+roffset),coffset:(128+coffset)]

	return data.astype('float32'), label, targetSerNum

def readMSTARDir(dirname):
	data = np.zeros([128*128,0],dtype = 'float32')
	labels = []
	serNums = []
	files = os.listdir(dirname)

	for f in files:
		fullpath = os.path.join(dirname,f)
		if os.path.isdir(fullpath):
			if 'SLICY' in f:
				continue
			d,l,sn = readMSTARDir(fullpath)
			data = np.concatenate((data,d),axis=1)
			labels = labels + l
			serNums = serNums + sn
		else:
			print(fullpath)
			if not fnmatch(f,'*.[0-9][0-9][0-9]'):
				continue
			d,l,sn = readMSTARFile(os.path.join(dirname,f))
			print(d.shape)
			data = np.concatenate((data,d.reshape(-1,1)),axis=1)
			labels = labels + [l]
			serNums = serNums + [sn]

	return data, labels, serNums

# def main1():
	# if len(sys.argv) < 3:
	# 	sys.exit()

	# filename = sys.argv[1]
	# outputfile = sys.argv[2]
def main1(filename, outputfile):
	data, labels, serNums = readMSTARDir(os.path.join(filename,'TRAIN'))

	mstar_dic_train = dict()

	mstar_dic_train['data'] = data
	mstar_dic_train['labels'] = labels
	mstar_dic_train['serial numbers'] = serNums

	data, labels, serNums = readMSTARDir(os.path.join(filename,'TEST'))

	mstar_dic_test = dict()

	mstar_dic_test['data'] = data
	mstar_dic_test['labels'] = labels
	mstar_dic_test['serial numbers'] = serNums

	labels = list(set(labels))

	label_dict = dict()

	for i in range(len(labels)):
		label_dict[labels[i]] = i

	for i in range(len(mstar_dic_train['labels'])):
		mstar_dic_train['labels'][i] = label_dict[mstar_dic_train['labels'][i]]

	for i in range(len(mstar_dic_test['labels'])):
		mstar_dic_test['labels'][i] = label_dict[mstar_dic_test['labels'][i]]


	f = open(os.path.join(outputfile,'data_batch_1'),'wb')
	pickle.dump(mstar_dic_train,f)

	f.close()

	f = open(os.path.join(outputfile,'test_batch'),'wb')
	pickle.dump(mstar_dic_test,f)

	f.close()

	meta_dic = dict()

	meta_dic['num_cases_per_batch'] = len(mstar_dic_train['labels'])
	meta_dic['label_names'] = labels

	f = open(os.path.join(outputfile,'batches.meta'),'wb')
	pickle.dump(meta_dic,f)

	f.close()

# def main2():
# 	if len(sys.argv) < 3:
# 		sys.exit()

# 	filename = sys.argv[1]
# 	outputfile = sys.argv[2]
def main2(filename, outputfile):
	data, labels, serNums = readMSTARDir(filename)

	mstar_dic_train = dict()

	mstar_dic_train['data'] = data
	mstar_dic_train['labels'] = labels
	mstar_dic_train['serial numbers'] = serNums

	labels = list(set(labels))

	label_dict = dict()

	for i in range(len(labels)):
		label_dict[labels[i]] = i

	for i in range(len(mstar_dic_train['labels'])):
		mstar_dic_train['labels'][i] = label_dict[mstar_dic_train['labels'][i]]

	f = open(os.path.join(outputfile,'data_batch_1'),'wb')
	pickle.dump(mstar_dic_train,f)

	f.close()

	meta_dic = dict()

	meta_dic['num_cases_per_batch'] = len(mstar_dic_train['labels'])
	meta_dic['label_names'] = labels

	f = open(os.path.join(outputfile,'batches.meta'),'wb')
	pickle.dump(meta_dic,f)

	f.close()

# def main3():
	# if len(sys.argv) < 3:
	# 	sys.exit()

	# filename = sys.argv[1]
	# outputfile = sys.argv[2]
def main3(filename, outputfile):
	data, labels, serNums = readMSTARDir(os.path.join(filename,'17_DEG'))

	mstar_dic_train = dict()

	mstar_dic_train['data'] = data
	mstar_dic_train['labels'] = labels
	mstar_dic_train['serial numbers'] = serNums

	data, labels, serNums = readMSTARDir(os.path.join(filename,'15_DEG'))

	mstar_dic_test = dict()

	mstar_dic_test['data'] = data
	mstar_dic_test['labels'] = labels
	mstar_dic_test['serial numbers'] = serNums

	labels = sorted(list(set(labels)))

	label_dict = dict()

	for i in range(len(labels)):
		label_dict[labels[i]] = i

	for i in range(len(mstar_dic_train['labels'])):
		mstar_dic_train['labels'][i] = label_dict[mstar_dic_train['labels'][i]]

	for i in range(len(mstar_dic_test['labels'])):
		mstar_dic_test['labels'][i] = label_dict[mstar_dic_test['labels'][i]]


	f = open(os.path.join(outputfile,'data_batch_1'),'wb')
	pickle.dump(mstar_dic_train,f)

	f.close()

	f = open(os.path.join(outputfile,'test_batch'),'wb')
	pickle.dump(mstar_dic_test,f)

	f.close()

	meta_dic = dict()

	meta_dic['num_cases_per_batch'] = len(mstar_dic_train['labels'])
	meta_dic['label_names'] = labels

	f = open(os.path.join(outputfile,'batches.meta'),'wb')
	pickle.dump(meta_dic,f)

	f.close()

def main4(filename, outputfile):
	data, labels, _ = readMSTARDir(os.path.join(filename,'data'))

	X_train, X_test, y_train, y_test = train_test_split(data.T, labels, test_size=0.2)
	
	mstar_dic_train = dict()

	mstar_dic_train['data'] = X_train.T
	mstar_dic_train['labels'] = y_train

	mstar_dic_test = dict()

	mstar_dic_test['data'] = X_test.T
	mstar_dic_test['labels'] = y_test

	labels = sorted(list(set(labels)))

	label_dict = dict()

	for i in range(len(labels)):
		label_dict[labels[i]] = i

	for i in range(len(mstar_dic_train['labels'])):
		mstar_dic_train['labels'][i] = label_dict[mstar_dic_train['labels'][i]]

	for i in range(len(mstar_dic_test['labels'])):
		mstar_dic_test['labels'][i] = label_dict[mstar_dic_test['labels'][i]]


	f = open(os.path.join(outputfile,'data_batch_1'),'wb')
	pickle.dump(mstar_dic_train,f)

	f.close()

	f = open(os.path.join(outputfile,'test_batch'),'wb')
	pickle.dump(mstar_dic_test,f)

	f.close()

	meta_dic = dict()

	meta_dic['num_cases_per_batch'] = len(mstar_dic_train['labels'])
	meta_dic['label_names'] = labels

	f = open(os.path.join(outputfile,'batches.meta'),'wb')
	pickle.dump(meta_dic,f)

	f.close()

if __name__ == '__main__':
	#main3()
	# main3("D:\My Documents\Work\Aerospace\MSTAR_tensorflow\MSTAR_PUBLIC_MIXED_TARGETS_BOTH", "D:\My Documents\Work\Aerospace\MSTAR_tensorflow\output")
	#main2("D:\My Documents\Work\Aerospace\MSTAR_tensorflow\MSTAR_PUBLIC_MIXED_TARGETS_BOTH", "D:\My Documents\Work\Aerospace\MSTAR_tensorflow\output")
	#main1("D:\My Documents\Work\Aerospace\MSTAR_tensorflow\MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY\TARGETS", "D:\My Documents\Work\Aerospace\MSTAR_tensorflow\output")
	#main4("D:\My Documents\Work\Aerospace\MSTAR_tensorflow\MSTAR_PUBLIC_MIXED_TARGETS_BOTH", "D:\My Documents\Work\Aerospace\MSTAR_tensorflow\output")
	main4("D:\My Documents\Work\Aerospace\MSTAR_tensorflow\MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY", "D:\My Documents\Work\Aerospace\MSTAR_tensorflow\output")
# print phoenix_header

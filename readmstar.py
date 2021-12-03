import numpy as np
import pickle
import os
from fnmatch import fnmatch
from sklearn.model_selection import train_test_split

def readMSTARFile(filename):
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

def main(filename, outputfile):
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
	#main("D:\My Documents\Work\Aerospace\MSTAR_tensorflow\MSTAR_PUBLIC_MIXED_TARGETS_BOTH", "D:\My Documents\Work\Aerospace\MSTAR_tensorflow\output")
	#main("D:\My Documents\Work\Aerospace\MSTAR_tensorflow\MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY", "D:\My Documents\Work\Aerospace\MSTAR_tensorflow\output")
	main("D:\My Documents\Work\Aerospace\MSTAR_tensorflow\MSTAR_ALL_TARGETS", "D:\My Documents\Work\Aerospace\MSTAR_tensorflow\output")

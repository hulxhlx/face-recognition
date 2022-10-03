#caculate distance,get val,far,accuracy,draw the tread-off, roc cruve and implement error analysis

import numpy as np
import facenet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
from sklearn.metrics import roc_curve  # 返回fpr、tpr、threshhold
from sklearn.metrics import roc_auc_score  # 返回ROC曲线下的面积
from sklearn.metrics import auc  # 返回ROC曲线下的面积
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import KFold
from scipy import interpolate

def calculate_val_far(threshold, dist, actual_issame):
	predict_issame = np.less(dist, threshold)
	true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
	false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
	n_same = np.sum(actual_issame)
	n_diff = np.sum(np.logical_not(actual_issame))
	val = float(true_accept) / (float(n_same) + 1)
	far = float(false_accept) / (float(n_diff) + 1)  # change
	return val, far


def calculate_val(thresholds, dist, actual_issame, far_target, nrof_folds=10,
                  distance_metric=0,  subtract_mean=False):

	nrof_pairs = len(actual_issame)
	nrof_thresholds = len(thresholds)
	k_fold = KFold(n_splits=nrof_folds, shuffle=False)

	val = np.zeros(nrof_folds)
	far = np.zeros(nrof_folds)

	indices = np.arange(nrof_pairs)

	for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
		# Find the threshold that gives FAR = far_target
		far_train = np.zeros(nrof_thresholds)
		for threshold_idx, threshold in enumerate(thresholds):
			_, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
		if np.max(far_train) >= far_target:
			f = interpolate.interp1d(far_train, thresholds, kind='slinear')
			threshold = f(far_target)
		else:
			threshold = 0.0

		val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

	val_mean = np.mean(val)
	far_mean = np.mean(far)
	val_std = np.std(val)
	return val_mean, val_std, far_mean




def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far

def calculate_accuracy(threshold, dist, actual_issame):
	predict_issame = np.less(dist, threshold)
	tp = np.sum(np.logical_and(predict_issame, actual_issame))
	fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
	tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
	fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
	tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
	fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
	acc = float(tp + tn) / dist.size
	return tpr, fpr, acc

def totalcalculate(threshold, dist, actual_issame):
	predict_issame = np.less(dist, threshold)
	tp = np.sum(np.logical_and(predict_issame, actual_issame))
	fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
	tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
	fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
	C2 = confusion_matrix(actual_issame, predict_issame, labels=[True, False])

	def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0,
	                  subtract_mean=False):
		assert (embeddings1.shape[0] == embeddings2.shape[0])
		assert (embeddings1.shape[1] == embeddings2.shape[1])
		nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
		nrof_thresholds = len(thresholds)
		k_fold = KFold(n_splits=nrof_folds, shuffle=False)

		val = np.zeros(nrof_folds)
		far = np.zeros(nrof_folds)

		indices = np.arange(nrof_pairs)

		for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
			if subtract_mean:
				mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
			else:
				mean = 0.0
			dist = distance(embeddings1 - mean, embeddings2 - mean)

			# Find the threshold that gives FAR = far_target
			far_train = np.zeros(nrof_thresholds)
			for threshold_idx, threshold in enumerate(thresholds):
				_, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
			if np.max(far_train) >= far_target:
				f = interpolate.interp1d(far_train, thresholds, kind='slinear')
				threshold = f(far_target)
			else:
				threshold = 0.0

			val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

		val_mean = np.mean(val)
		far_mean = np.mean(far)
		val_std = np.std(val)
		return val_mean, val_std, far_mean

	def calculate_val_far(threshold, dist, actual_issame):
		predict_issame = np.less(dist, threshold)
		true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
		false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
		n_same = np.sum(actual_issame)
		n_diff = np.sum(np.logical_not(actual_issame))
		val = float(true_accept) / (float(n_same) + 1)
		far = float(false_accept) / (float(n_diff) + 1)  # change
		return val, far


	sns.heatmap(C2, annot=True)
	plt.xlabel('Pred')
	plt.ylabel('True')
	plt.show()

	tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
	print('tpr,fpr,precision,recall,f1score')
	fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
	acc = float(tp + tn) / dist.size
	Precision = tp / (tp + fp)

	Recall = tp / (tp + fn)


	F1score = 2 * Precision * Recall /(Precision + Recall)
	print(tpr,fpr,Precision,F1score)
	# fpr1, tpr1, thresholds1 = roc_curve(actual_issame, dist, pos_label=2)
	# print(roc_auc_score(actual_issame, dist))
	# plt.plot(fpr1, tpr1, c='r', lw=2, alpha=0.7)
	# plt.show()
	return tpr, fpr, acc

def distance(embeddings1, embeddings2):
	diff = np.subtract(embeddings1, embeddings2)
	dist = np.sum(np.square(diff), 1)
	return dist
pathlist = np.load('pathlist.npy')
print(pathlist.shape)
lab = np.load('lab.npy')

embeddings = np.load('embseq.npy')
actual_issame = np.load('issame_list.npy')
print(actual_issame.shape)#2400，
thresholds = np.arange(0, 4, 0.01)
embeddings1 = embeddings[0::2]
embeddings2 = embeddings[1::2]
nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
nrof_thresholds = len(thresholds)

# dist = distance(embeddings1, embeddings2 )
dist = np.load('bestdist.npy')
acc_train = np.zeros((nrof_thresholds))
thresholds = np.arange(0, 4, 0.01)
tprl= []
fprl = []
for threshold_idx, threshold in enumerate(thresholds):
	tpr_, fpr_, acc_train[threshold_idx] = calculate_accuracy(threshold, dist, actual_issame)
	tprl.append(tpr_)
	fprl.append(fpr_)

plt.title('roc curve')
plt.plot(fprl, tprl, c='r', lw=2, alpha=0.7)
plt.show()
print(auc(fprl, tprl))
best_threshold_index = np.argmax(acc_train)
print(best_threshold_index)


tpr, fpr, accuracy = totalcalculate(thresholds[best_threshold_index], dist,
		                                              actual_issame)


print(thresholds[best_threshold_index])#0.93
# print(max(dist))#2.4
nomatchdis = []
matchdis=[]
w=0
for dis in dist:
	if actual_issame[w] == True:
		matchdis.append(dis)
	else:
		nomatchdis.append(dis)
	w=w+1
#
plt.xlabel('distance')
plt.ylabel('number of distance')
plt.yticks(range(0,5))
sns.set_palette("hls") #draw tread off
x=[thresholds[best_threshold_index],thresholds[best_threshold_index]]
y = [0,3]
plt.plot(x,y)
sns.distplot(matchdis,color="r",bins=300,kde=True)
sns.distplot(nomatchdis,color="g",bins=300,kde=True)

plt.show()

val, val_std, far = calculate_val(thresholds, dist,
	                                              np.asarray(actual_issame), 1e-3, nrof_folds=10,
	                                              distance_metric=0, subtract_mean=False)
print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
w=0
#find mismatch label
fr_list=[]#true but predict false
fa_list=[]
for dis in dist:
	if actual_issame[w] == True and dis > 0.93:
		fr_list.append(w)
	if actual_issame[w] == False and dis < 0.93:
		fa_list.append(w)
	w=w+1
print(fr_list)
#inception:[5, 17, 29, 580, 585, 916]
#seq:[14, 19, 22, 275, 283, 361, 363, 369, 909, 911, 914, 915, 920, 922, 923, 928, 929, 948, 1053, 1057]
print(fa_list)
#[1200, 1202, 1210, 1217, 1219, 1220, 1221, 1255, 1265, 1282, 1286, 1288, 1294, 1297, 1299, 1315, 1332, 1339, 1356, 1359, 1360, 1367, 1401, 1404, 1410, 1416, 1441, 1452, 1454, 1460, 1462, 1473, 1475, 1495, 1499, 1502, 1519, 1526, 1528, 1531, 1538
#1540, 1546, 1549, 1555, 1562, 1564, 1573, 1576, 1596, 1605, 1608, 1609, 1622, 1623, 1632, 1640, 1667, 1669, 1677, 1688, 1704, 1721, 1753, 1768, 1785, 1808, 1825, 1867, 1873, 1898, 1916, 1919, 1927, 1950, 1954, 1967, 2005, 2007, 2012, 2019, 2023,
#2048, 2052, 2059, 2061, 2066, 2077, 2096, 2124, 2155, 2157, 2162, 2165, 2171, 2181, 2187, 2202, 2224, 2229, 2230, 2234, 2235, 2249, 2259, 2264, 2265, 2283, 2314, 2323, 2332, 2349, 2359, 2365, 2380]
#[1276, 1393, 1404, 1473, 1502, 1538, 1687, 1690, 1733, 1804, 1818, 1851, 1913, 1919, 1921, 1949, 2034, 2042, 2050, 2178, 2213, 2332]
# #
#printing common error list
commonerror_List=[1404,1473,1502,1538,1919,2332]
fig, axs = plt.subplots(len(commonerror_List), 2, figsize=(2,len(commonerror_List)*1.5 ))

for i in range(len(commonerror_List)):
	img = cv2.imread(pathlist[commonerror_List[i]*2])
	axs[i, 0].imshow(img)
	# axs[i, 0].set_title(
	#                     f' distance: {"%.3f" % dist[commonerror_List[i]]}'
	#                     , fontsize=8)
	axs[i, 0].axis('off')
	img = cv2.imread(pathlist[commonerror_List[i]*2+1])
	axs[i, 1].imshow(img)
	axs[i, 1].axis('off')

plt.show()
plt.close()



fig, axs = plt.subplots(len(fr_list), 2, figsize=(2,len(fr_list)*1.5 ))

for i in range(len(fr_list)):
	img = cv2.imread(pathlist[fr_list[i]*2])
	axs[i, 0].imshow(img)
	axs[i, 0].set_title(
	                    f' distance: {"%.3f" % dist[fr_list[i]]}'
	                    , fontsize=8)
	axs[i, 0].axis('off')
	img = cv2.imread(pathlist[fr_list[i]*2+1])
	axs[i, 1].imshow(img)
	axs[i, 1].axis('off')

plt.show()
plt.close()

fig, axs = plt.subplots(len(fa_list), 2, figsize=(2,len(fa_list)*1.5 ))

for i in range(len(fa_list)):
	img = cv2.imread(pathlist[fa_list[i]*2])
	axs[i, 0].imshow(img)
	axs[i, 0].set_title(
	                    f' distance: {"%.3f" % dist[fa_list[i]]}'
	                    , fontsize=8)
	axs[i, 0].axis('off')
	img = cv2.imread(pathlist[fa_list[i]*2+1])
	axs[i, 1].imshow(img)
	axs[i, 1].axis('off')

plt.show()
plt.close()












#use pca and tsne to draw a picture for everyone's embedding.7
#use svm to classify

import numpy as np
import facenet
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.manifold import TSNE
from sklearn import decomposition
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression

emb40 = np.load('embseq40.npy')
label = np.load('lab40.npy')

print(emb40[1].shape)

pca = decomposition.PCA(n_components=50)
emb40_reduced = pca.fit_transform(emb40)
tsne = TSNE(n_components=2)
emb40_reduced_tsne = tsne.fit_transform(emb40_reduced)
color=[]
for i in range(len(label)):
	color.append([[i,i]])

plt.scatter( emb40_reduced_tsne[:200, 0], emb40_reduced_tsne[:200, 1], c=label[:200],marker='o', cmap='tab20b' )
plt.scatter( emb40_reduced_tsne[200:, 0], emb40_reduced_tsne[200:, 1], c=label[200:],marker='v', cmap='tab20b' )

plt.show()


clf = LinearSVC(C=0.001)
scoring = ['accuracy']
scores1 = cross_validate(clf,emb40, label,scoring=scoring, cv=10)
print(scores1['test_accuracy'].mean())
print(scores1['test_accuracy'].std())


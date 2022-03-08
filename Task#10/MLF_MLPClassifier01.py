#!/usr/bin/env python
# coding: utf-8

# ## Листинг программы MLF_MLPClassifier001

# In[43]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
X_D2, y_D2 = make_blobs(n_samples = 200, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
plt.figure()
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker= 'o', s=50, cmap=cmap_bold)
plt.show()

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)


clf = MLPClassifier(hidden_layer_sizes = [10, 10], alpha = 5,
                   random_state = 0, solver='lbfgs').fit(X_train, y_train)


print('Blob dataset')
print('Accuracy of NN classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# ### Листинг программы MLF_MLPClassifier001_t1

# In[1]:


#MLF_MLPClassifier001_t1
get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
X_D2, y_D2 = make_blobs(n_samples = 200, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
plt.figure()
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker= 'o', s=50, cmap=cmap_bold)
plt.show()

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#Metrics
from sklearn.metrics import confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)


clf = MLPClassifier(hidden_layer_sizes = [10, 10], alpha = 5,
                   random_state = 0, solver='lbfgs').fit(X_train, y_train)

predictions=clf.predict(X_test)

print('Accuracy of NN classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
#task01
print(classification_report(y_test,predictions))
matrix = confusion_matrix(y_test, predictions)#,labels)
print('Confusion matrix on test set\n',matrix)


# ### MLF_MLPClassifier001_t1+plot_class_regions_for_classifier

# In[2]:


#MLF_MLPClassifier001_t1
get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
X_D2, y_D2 = make_blobs(n_samples = 200, n_features = 2, centers = 8, cluster_std = 1.3, random_state = 4)
plt.figure()
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker= 'o', s=50, cmap=cmap_bold)
plt.show()

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#Metrics
from sklearn.metrics import confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)


clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 0.1,
                   random_state = 0, solver='lbfgs').fit(X_train, y_train)

predictions=clf.predict(X_test)

print('Accuracy of NN classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
#task01
print(classification_report(y_test,predictions))
matrix = confusion_matrix(y_test, predictions)#,labels)
print('Confusion matrix on test set\n',matrix)



#X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

#nnclf = MLPClassifier(hidden_layer_sizes = [10, 10], solver='lbfgs',
#                     random_state = 0).fit(X_train, y_train)

from adspy_shared_utilities import plot_class_regions_for_classifier
plot_class_regions_for_classifier(clf, X_train, y_train, X_test, y_test,
                                 'Dataset 1: Neural net classifier, 2 layers, 10/10 units')


# In[ ]:





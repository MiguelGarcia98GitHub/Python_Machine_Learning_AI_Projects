import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer


# * Load the data
data = load_breast_cancer()

# * Print the input properties (or features) 
print(data.feature_names)

# * Print the output possible properties
print(data.target_names)

# * Convert our data to Numpy arrays so that we can process it
X = np.array(data.data)
Y = np.array(data.target)

# * Divide data for training and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)

# * Create our model using K-Nearest-Neighbours algorithm, we give a value for neighbours of 5, we can try different values, but odd values like 5, 7, 9... are recommended for most cases
knn = KNeighborsClassifier(n_neighbors=5)

# * Train our model
knn.fit(X_train, Y_train)

# * Test our model using the test data to see how it performs, and print the value. a 0.90 would equal a 90% success rate for example, which is reasonably good. 
accuracy = knn.score(X_test, Y_test)
print(accuracy)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# * Read the CSV file
data = pd.read_csv('student-mat.csv', sep=';')


# * Get the data that we decided that will be relevant for our analysis
data = data[['age', 'sex', 'studytime', 'absences', 'G1', 'G2', 'G3']]


# * Sex is stored as F or M, but this wont work, we need it to be numbers (so we change it)
data['sex'] = data['sex'].map({'F': 0, 'M': 1})
print(data)


# * Now we choose a label (a desired label that we want, no specific reason) and make it a variable to make it easier to work with
prediction = 'G3'


# * Reformat our data to Numpy arrays, since SKLearn does not accept pandas dataframes 
X = np.array(data.drop([prediction], axis=1))
Y = np.array(data[prediction])


# * Split our data into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


# * Create our model
model = LinearRegression()


# * Train our model
model.fit(X_train, Y_train)


# * Test our model with the test data
accuracy = model.score(X_test, Y_test)


# * Check the test result: a 0.90 would equal a 90% success rate for example, which is reasonably good. 
# * The splitting of training and test data is always random so we should have slightly different results on each run
print(accuracy)


# * Now that we know that our model is trained and is somewhat reliable, lets do some additional tests by passing it some new data (so we can predict is accuracy predicting the final grade of given student)
X_new = np.array([[18, 1, 3, 40, 15, 16]])
Y_new = model.predict(X_new)
print(Y_new)

# * Visualize some interesting chart for our data, showing a clear representation of the correlation between the second grade and the final grade
plt.scatter(data['G2'], data['G3'])
plt.title('Correlation between Second Grade and Final Grade')
plt.xlabel('Second Grade')
plt.ylabel('Final Grade')
plt.show()
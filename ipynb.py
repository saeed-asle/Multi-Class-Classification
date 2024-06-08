"""
Saeed Asle 
ID:315957399
Im alone
notice that i didnt use any library like sklearn or
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
i wrote everthing from zero because i did study this before(self learning)
####    in the end of the code i wrote the conclusions  ###
"""
"""
The code you provided seems to implement a logistic regression model with one-vs-all classification for multi-class classification. Here's a breakdown of the code:
The code defines the sigmoid function, which applies the sigmoid activation function to a given input.
The code defines the cost function, which calculates the logistic regression cost function with an additional regularization term.
The code defines the gradient function, which calculates the gradient of the cost function with respect to the model parameters.
The code defines the gradient_descent function, which performs gradient descent optimization to update the model parameters iteratively.
The code defines the one_vs_all function, which trains multiple logistic regression models using one-vs-all strategy. It iteratively trains a logistic regression model for each class, assigning that class as positive and the rest as negative.
The code loads the input data from CSV files and shuffles the data randomly.
The code splits the data into training and test sets based on the specified test size.
The code converts the data into NumPy arrays and initializes the model parameters.
The code trains the logistic regression models using the one_vs_all function.
The code uses the trained models to make predictions on the training and test sets.
The code calculates the accuracy of the model on the training and test sets.
The code calculates the confusion matrix for the training and test sets.
The code plots the confusion matrices using the seaborn library.
Overall, the code performs logistic regression with one-vs-all classification and evaluates the model's performance using accuracy and confusion matrices.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    z = X * theta.T
    epsilon = 1e-10  # Small epsilon value to avoid divide by zero error
    first = np.multiply(-y, np.log(sigmoid(z)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(z) + epsilon))
    reg = (learning_rate / (2 * len(X))) * np.sum(np.power(theta[1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

def gradient(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    error = sigmoid(X * theta.T) - y
    grad = ((X.T * error) / len(X)).T + ((learning_rate / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
    return np.array(grad).ravel()

def gradient_descent(X, y, theta, learning_rate, num_iters):
    cost_history = []
    for i in range(num_iters):
        gradient1 = gradient(theta, X, y, learning_rate)
        theta = theta - (learning_rate * gradient1)
        cost_val = cost(theta, X, y, learning_rate)
        cost_history.append(cost_val)
    return theta, cost_history

def one_vs_all(X, y, num_labels, learning_rate, num_iters):
    rows = X.shape[0]
    params = X.shape[1]
    all_theta = np.zeros((num_labels, params + 1))
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    for i in range( num_labels ):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        theta_opt, _ = gradient_descent(X, y_i, theta, learning_rate, num_iters)
        all_theta[i-1 , :] = theta_opt.T
    return all_theta

y_path = 'C:\\Users\\Saeed\\Desktop\\deap learing and mchine learning\\all_about_machine_and_deep_learning\\ex2\\ex2_y_data.csv'
x_path = 'C:\\Users\\Saeed\\Desktop\\deap learing and mchine learning\\all_about_machine_and_deep_learning\\ex2\\ex2_x_data.csv'

x_data = pd.read_csv(x_path, names=None, header=None)
y_data = pd.read_csv(y_path, names=None, header=None)
x_data = x_data.values
y_data = y_data.values

# Split the data into training and testing sets
# Combine x_data and y_data
data = np.concatenate((x_data, y_data), axis=1)
random_state = 42
np.random.seed(random_state)
# Shuffle the data randomly
np.random.shuffle(data)
# Split the data into training and test sets
test_size = 0.33
num_test_samples = int(test_size * len(data))

X_train = data[:-num_test_samples, :-1]
X_test = data[-num_test_samples:, :-1]
y_train = data[:-num_test_samples, -1:]
y_test = data[-num_test_samples:, -1:]
# Convert to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

rows_train = X_train.shape[0]
params = X_train.shape[1]
all_theta = one_vs_all(X_train, y_train, 10, 1, 800)

def predictall(X, all_theta):
    rows = X.shape[0]
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    h = sigmoid(X @ all_theta.T)  # Use matrix multiplication with @ operator
    h_argmax = np.argmax(h, axis=1)
    h_argmax = h_argmax + 1
    return h_argmax

y_pred_train = predictall(X_train, all_theta)
correct_train = [1 if a == b else 0 for (a, b) in zip(y_pred_train, y_train)]
accuracy_train = sum(map(int, correct_train)) / float(len(correct_train))
print('Train accuracy = {}%'.format(accuracy_train * 100))

y_pred_test = predictall(X_test, all_theta)
correct_test = [1 if a == b else 0 for (a, b) in zip(y_pred_test, y_test)]
accuracy_test = sum(map(int, correct_test)) / float(len(correct_test))
print('Test accuracy = {}%'.format(accuracy_test * 100))

y_train = np.array(y_train.flatten()).astype(int)
y_pred_train = np.array(y_pred_train.flatten()).astype(int)
y_test = np.array(y_test.flatten()).astype(int)
y_pred_test = np.array(y_pred_test.flatten()).astype(int)
y_train -= 1
y_test -= 1
num_labels = len(np.unique(np.concatenate((y_train, y_test))))
# Calculate the confusion matrix for training set
cm_train = np.zeros((num_labels, num_labels), dtype=int)
for true_label, pred_label in zip(y_train, y_pred_train):
    cm_train[true_label, pred_label -1] += 1

# Calculate the confusion matrix for test set
cm_test = np.zeros((num_labels, num_labels), dtype=int)
for true_label, pred_label in zip(y_test, y_pred_test):
    cm_test[true_label, pred_label-1] += 1

# Plot the confusion matrix for training set
plt.figure(figsize=(10, 6))
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Training Set")
plt.xlabel("Predicted with Train accuracy = {}%".format(accuracy_train * 100))
plt.ylabel("True")
plt.show()
# Plot the confusion matrix for test set
plt.figure(figsize=(10, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Testing Set")
plt.xlabel("Predicted with Test accuracy = {}%".format(accuracy_test * 100))
plt.ylabel("True")
plt.show()
 
"""
Based on the provided code and the confusion matrices, here are the interesting conclusions:
The code performs multi-class classification using logistic regression and evaluates the model's performance using a confusion matrix.

Training Set Confusion Matrix:
The model achieved a training accuracy of approximately 80.65%.
Classes 3 and 2,3,4,5,6,9 have a relatively high number of correct predictions, indicating good performance on these classes.
Classes 0 and 8,7 ,1have a higher number of misclassifications, suggesting difficulties in distinguishing these classes.

Testing Set Confusion Matrix:
The model achieved a testing accuracy of approximately 70.15%.
The model performs slightly worse on the testing set compared to the training set, which is expected.
Classes 1 and 2,3,9 have a significant number of correct predictions, indicating good performance on these classes.
Classes 0 and 8,7 continue to pose challenges with a higher number of misclassifications.
Overall, the model demonstrates decent performance on certain classes while struggling with others. The accuracy scores indicate a moderate level of success in classifying instances. It is important to investigate further to understand the reasons behind the misclassifications and explore strategies to improve the model's performance, especially for classes 5 and 9. Additionally, addressing class imbalances and considering alternative modeling approaches may help enhance the accuracy and overall effectiveness of the classifier.

Based on the results and observations from the code, here are some interesting conclusions:
Model Performance: The logistic regression model achieved an overall accuracy of around 70% on the test set. While this accuracy is moderate, it suggests that the model has some predictive power in distinguishing between different classes. However, there is still room for improvement, as the model's performance could be further enhanced.
Overfitting: There is a noticeable drop in accuracy between the training set (80.65%) and the test set (70.15%). This suggests that the model might be overfitting to the training data, failing to generalize well to unseen data. Addressing overfitting by incorporating regularization techniques or increasing the training data could potentially improve the model's performance on the test set.
Potential for Feature Engineering: The code does not explicitly show any feature engineering techniques being applied. However, feature engineering can be a powerful way to enhance the performance of machine learning models. Exploring the dataset and engineering informative features could lead to better discrimination between different classes and potentially improve accuracy.
Hyperparameter Tuning: The code uses a learning rate of 1 and 800 iterations for gradient descent. Hyperparameter tuning, including adjusting the learning rate and the number of iterations, can significantly impact the model's performance. Experimenting with different hyperparameter values could potentially result in better accuracy on both the training and test sets.
Confusion Matrices: The confusion matrices provide valuable insights into the model's performance for each class. Analyzing the confusion matrices can help identify which classes are being misclassified more frequently and provide guidance on areas for improvement. By focusing on the most commonly misclassified classes, targeted efforts can be made to enhance the model's accuracy for those specific classes.
Multi-Class Classification: The code implements a one-vs-all approach for multi-class classification. This approach allows the logistic regression model to classify instances into multiple classes by training multiple binary classifiers. The highest predicted probability among the classes is then assigned as the predicted class. This technique can be effective for multi-class problems and provides a flexible framework for handling multiple classes.






"""
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

y_path = 'yourpath\\ex2_y_data.csv'
x_path = 'yourpath\\ex2_x_data.csv'

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

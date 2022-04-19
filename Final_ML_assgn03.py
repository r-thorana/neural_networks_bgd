import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######################load the data###########################
X = np.loadtxt(open("./data/X.csv", "rb"), delimiter=",")
Y_o = np.loadtxt(open("./data/Y.csv", "rb"), delimiter=",")
m = X.shape[0]
n = Y_o.shape[0]
eta = 0.2
# W1 = np.loadtxt(open("./data/W1.csv", "rb"), delimiter=",")
# W2 = np.loadtxt(open("./data/W2.csv", "rb"), delimiter=",")
lst = []
for value in Y_o:
    values = np.zeros(10)
    values[int(value)-1] = 1
    lst.append(values)
Y = np.array(lst)
# Y = pd.get_dummies(Y.flatten())
W1 = np.loadtxt(open("./data/initial_W1.csv", "rb"), delimiter=",")
W2 = np.loadtxt(open("./data/initial_W2.csv", "rb"), delimiter=",")

def sigmoid(z):
    return 1/(1+np.exp(-z))

def logsigmoid(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))

#############################loss function#################################################
def cost(Y, Y_cap, m, lmbda, W1, W2 ):
    temp1 = np.multiply(Y, np.log(Y_cap))
    temp2 = np.multiply(1-Y, np.log(1-Y_cap))
    temp3 = np.sum(temp1 + temp2)
    temp4 = np.sum(temp3/(-m))
    # print(temp4)
    sum1 = np.sum(np.sum(np.power(W1[:, 1:], 2), axis=1))
    sum2 = np.sum(np.sum(np.power(W2[:, 1:], 2), axis=1))
    second_term = (sum1 + sum2) * lmbda / (2 * m)
    # print(second_term)
    return temp4 + second_term

def batchgradientdescent(X,Y,W1,W2,eta):
    m = X.shape[0]
    n = (Y.shape[1])
    lmbda = 3
    k=0
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    t = X.shape[1]
    costHistory = []
    while k < 500:
        Z1 = X @ W1.T
        H = sigmoid(Z1)
        H = np.concatenate((np.ones((m,1)), H), axis=1 )
        p = H.shape[1]
        Z2 = np.dot(H, W2.T)
        Y_cap = sigmoid(Z2)
        max_index_row = np.argmax(Y_cap, axis=1)
        pred_y = (max_index_row + 1)
        costHistory.append(cost(Y, Y_cap, m, lmbda, W1, W2))
        # print(costHistory)
        final_pargradW2 = np.zeros((n,p))
        final_pargradW1 = np.zeros((p-1, t))

        for i in range(m):
            #Vector -- 1 * 10
            beta2 = Y_cap[i,:] - Y[i,:]
            beta1 = (beta2 @ W2[:,1:26]) * logsigmoid(Z1[i, :])
            paragradW2 = (beta2.reshape(-1, 1)) * H[i, :]
            paragradW1 = (beta1.reshape(-1, 1)) * X[i, :]
            final_pargradW2 = final_pargradW2 + paragradW2
            final_pargradW1 = final_pargradW1 + paragradW1
        W1_new = W1.copy()
        W2_new = W2.copy()
        W2_new[:, 0] = 0  # 10 * 26
        W1_new[:,0] = 0  # 25 * 401
        gradW2 = ((1 / m) * final_pargradW2) + ((lmbda / m) * (W2_new))
        gradW1 = ((1 / m) * final_pargradW1) + ((lmbda / m) *(W1_new))
        W2 = W2 - (eta * gradW2)
        W1 = W1 - (eta * gradW1)
        k = k +1
    return costHistory,k, W1, W2,Y_cap,pred_y

cost, k, W1, W2, Y_cap, pred_y = batchgradientdescent(X, Y, W1, W2,eta)
print(cost)

#######################plotting graph################################################
plt.plot(cost)
plt.ylabel('cost')
plt.show()

###########################calculating Accuracy#####################################
c = 0
for i in range(len(Y_o)):
    if Y_o[i] == pred_y[i]:
        c = c + 1
    accuracy = (c/m) * 100
print("accuracy is :", accuracy)

############################ predicting results for given training examples ############################################
print("The predicted value of training example 2171 is:", pred_y[2170])
print("The predicted value of training example 145 is:", pred_y[144])
print("The predicted value of training example 1582 is:", pred_y[1581])
print("The predicted value of training example 2446 is:", pred_y[2445])
print("The predicted value of training example 3393 is:", pred_y[3392])
print("The predicted value of training example 815 is:", pred_y[814])
print("The predicted value of training example 1378 is:", pred_y[1377])
print("The predicted value of training example 529 is:", pred_y[528])
print("The predicted value of training example 3945  is:", pred_y[3944])
print("The predicted value of training example 4629 is:", pred_y[4628])





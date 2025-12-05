import numpy as np
import matplotlib.pyplot as plt

# Creating data set

# A
a =[0, 0, 1, 1, 0, 0,
   0, 1, 0, 0, 1, 0,
   1, 1, 1, 1, 1, 1,
   1, 0, 0, 0, 0, 1,
   1, 0, 0, 0, 0, 1]
# B
b =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0]
# C
c =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 1, 1, 1, 0]

# Creating labels
y =[[1, 0, 0],
   [0, 1, 0],
   [0, 0, 1]]

# Visualizing 
plt.imshow(np.array(a).reshape(5, 6))
plt.show()

# Convert into numpy array
x = [np.array(a).reshape(1, 30),  np.array(b).reshape(1, 30), 
                                np.array(c).reshape(1, 30)]

y = np.array(y)

print(x, "\n\n", y)


def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def feedForward(x, weights1, weights2):
    weightedInputs = x.dot(weights1)
    activationAppliedToWeightedInputs = sigmoid(weightedInputs)
    weightedInputs2 = activationAppliedToWeightedInputs.dot(weights2)
    activationAppliedToWeightedInputs2 = sigmoid(weightedInputs2)
    return activationAppliedToWeightedInputs2

def generateWeights(x, y):
    randomWeights = []
    for i in range(x * y):
        randomWeights.append(np.random.randn())
    return(np.array(randomWeights).reshape(x, y))

def loss(output, Y):
    return np.sum(np.square(output-Y))/len(y)

def backPropagation(x, y, weights1, weights2, alpha):
    



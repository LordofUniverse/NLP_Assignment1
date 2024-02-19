import numpy as np

np.random.seed(42)

"""
Sigmoid activation applied at each node.
"""
def sigmoid(x):
    # cap the data to avoid overflow?
    # x[x>100] = 100
    # x[x<-100] = -100
    # return 1/(1+np.exp(-x))
    return x**2

"""
Derivative of sigmoid activation applied at each node.
"""
def sigmoid_derivative(x):
    # return sigmoid(x)*(1-sigmoid(x))
    return 2*x

class NN:
    def __init__(self, input_dim, hidden_dim, activation_func = sigmoid, activation_derivative = sigmoid_derivative):
        """
        Parameters
        ----------
        input_dim : TYPE
            DESCRIPTION.
        hidden_dim : TYPE
            DESCRIPTION.
        activation_func : function, optional
            Any function that is to be used as activation function. The default is sigmoid.
        activation_derivative : function, optional
            The function to compute derivative of the activation function. The default is sigmoid_derivative.

        Returns
        -------
        None.

        """
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        # TODO: Initialize weights and biases for the hidden and output layers
        
        self.weights1 = np.random.normal(0, 1, (input_dim, hidden_dim))
        self.weights2 = np.random.normal(0, 1, (hidden_dim, 1))
        
        self.bias1 = np.random.normal(0, 1, (1, hidden_dim))
        self.bias2 = np.random.normal(0, 1, (1, 1))
        
    def forward(self, X):
        # Forward pass
        # TODO: Compute activations for all the nodes with the activation function applied 
        # for the hidden nodes, and the sigmoid function applied for the output node
        # TODO: Return: Output probabilities of shape (N, 1) where N is number of examples

        # print((X @ self.weights1).shape, self.bias1.shape)
        newX = self.activation_func(X @ self.weights1 + self.bias1)

        outputx = self.activation_func(newX @ self.weights2 + self.bias2)

        return outputx
    
    def backward(self, X, y, learning_rate):
        # Backpropagation
        # TODO: Compute gradients for the output layer after computing derivative of sigmoid-based binary cross-entropy loss
        # TODO: When computing the derivative of the cross-entropy loss, don't forget to divide the gradients by N (number of examples)  
        # TODO: Next, compute gradients for the hidden layer
        # TODO: Update weights and biases for the output layer with learning_rate applied
        # TODO: Update weights and biases for the hidden layer with learning_rate applied
        
        y = y[:,None]
        
        newX = X @ self.weights1 + self.bias1
        newXact = self.activation_func(X @ self.weights1 + self.bias1) # 1x2 matrix
        outputx = newXact @ self.weights2 + self.bias2
        ycap = self.activation_func(outputx)

        # weights2
        dldycap = - ((y / ycap) - (1-y)/(1-ycap))

        dycapdo = self.activation_derivative(outputx)

        dldw2 = np.mean(newXact * dldycap * dycapdo, axis=0).reshape(-1, 1)
        db2 = np.mean(dldycap * dycapdo, axis=0).reshape(-1, 1)

        ans = np.zeros((self.weights1.shape[1], X.shape[1]))
        for i in range(self.weights1.shape[1]):
            ans[i] = np.mean(X * self.activation_derivative(newX.T[i].reshape((-1, 1))) * dldycap * dycapdo * self.weights2[i], axis=0)

        db1 = np.mean(dldycap * dycapdo * self.weights2.flatten() * self.activation_derivative(newX), axis=0).reshape(1, -1)   

        self.weights2 -= learning_rate * dldw2
        self.weights1 -= learning_rate * ans.T
        
        self.bias2 -= learning_rate * db2
        self.bias1 -= learning_rate * db1

    def train(self, X, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            # Forward pass
            self.yhat = self.forward(X).reshape(-1,)

            # Backpropagation and gradient descent weight updates
            self.backward(X, y, learning_rate)
            # TODO: self.yhat should be an N times 1 vector containing the final
            # sigmoid output probabilities for all N training instances 
            # TODO: Compute and print the loss (uncomment the line below)
            loss = np.mean(-y*np.log(self.yhat) - (1-y)*np.log(1-self.yhat))
            # TODO: Compute the training accuracy (uncomment the line below)
            accuracy = np.mean((self.yhat > 0.5).reshape(-1,) == y)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            self.pred('pred_train.txt')

            if accuracy == 1:
                print("STOPPING")
                break
            
    def pred(self,file_name='pred.txt'):
        pred = self.yhat > 0.5
        with open(file_name,'w') as f:
            for i in range(len(pred)):
                f.write(str(self.yhat[i]) + ' ' + str(int(pred[i])) + '\n')

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data.csv"
    # eval_file_path = "data_eval.csv"
    
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    # np.random.shuffle(data)
    data1 = data[data[:, -1] == 1]
    data0 = data[data[:, -1] == 0]

    # print(data1.shape, data0.shape)

    data1_eval = data1[1024:, :]
    data1 = data1[:1024, :]
    data0_eval = data0[1024:, :]
    data0 = data0[:1024, :]

    data = np.concatenate((data1, data0))
    data_eval = np.concatenate((data1_eval, data0_eval))

    np.random.shuffle(data)
    np.random.shuffle(data_eval)

    # data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)
    # Separate the data into X (features) and y (target) arrays
    
    X = data[:, :-1]
    X_eval = data_eval[:, :-1]
    y = data[:, -1]
    y_eval = data_eval[:, -1]
    
    print(X)
    print(y)
    print(X_eval)
    print(y_eval)

    # Create and train the neural network
    # input_dim = X.shape[1]
    # hidden_dim = 4
    input_dim = 10
    hidden_dim = 5
    learning_rate = 0.05
    num_epochs = 100
    
    model = NN(input_dim, hidden_dim)
    model.train(X, y, learning_rate, num_epochs) #trained on concentric circle data 

    test_preds = model.forward(X_eval)
    model.pred('pred_eval.txt')

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Test accuracy: {test_accuracy:.4f}")

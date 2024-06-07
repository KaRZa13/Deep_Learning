import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn .metrics import accuracy_score

# Le graph t'as vu ;)
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print(f"Dimensions de X : {X.shape}")
print(f"Dimensions de y : {y.shape}")

# Voir le graph
plt.scatter(X[:,0], X[:, 1], c=y, cmap="winter")
plt.show()



def init(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

# Modèle d'apprentissage 
def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

# Difference entre le résultat attendu et le résultat obtenu
def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

# Calculs mathétiques de zinzin j'ai pas encore bien compris mais ça marche 
def gradiant(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return(dW, db)

# Aprrentissage parce que le neuronne il est pas con
def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return(W, b)

# Predictiction de la réponse 
def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5

#Et BOOM ! Le neuronne !
def artificial_neuron(X, y, learning_rate=0.1, n_iter=100):
    W, b = init(X)

    Loss = []

    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradiant(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    y_pred = predict(X, W, b)
    print(accuracy_score(y, y_pred))

    plt.plot(Loss)
    plt.show()

    return(W, b)


W, b = artificial_neuron(X, y)
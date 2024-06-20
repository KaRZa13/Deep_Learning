import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn .metrics import accuracy_score
import plotly.graph_objects as go

# Création du Data set
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print(f"Dimensions de X : {X.shape}")
print(f"Dimensions de y : {y.shape}")


plt.scatter(X[:,0], X[:, 1], c=y, cmap="winter")
plt.show()


def init(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

# Modèle d'apprentissage avec Z égal au produit matriciel de X et W plus le prm b 
def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A # Vecteur (100, 1) dans cet exemple

def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

# Formule de descente de gradiant (pas encore bien compris)
def gradiant(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return(dW, db)

# Apprentissage
def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return(W, b)

# Predictiction de la réponse 
def predict(X, W, b):
    A = model(X, W, b)
    #print(A)
    return A >= 0.5

# Assemblage du neuronne
def artificial_neuron(X, y, learning_rate=0.1, n_iter=100):
    #Initialisation de W et b
    W, b = init(X)

    history = []
    Loss = []

    # Boucle d'apprentissage
    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradiant(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)
        history.append([W, b, Loss, i])


    y_pred = predict(X, W, b)
    # Affichage de la performance du modède /1
    print(accuracy_score(y, y_pred))

    # Affichage de la courbe d'apprentissage avec le nombre d'érreur
    plt.plot(Loss)
    plt.show()

    return W, b

W, b = artificial_neuron(X, y)

print(W, b)

# Calcul de la droite de frontière de décision
x0 = np.linspace(-1, 4, 100)
x1 = (-W[0] * x0 - b) / W[1]

plt.scatter(X[:,0], X[:, 1], c=y, cmap="winter")
plt.plot(x0, x1, c="orange", lw=3)
plt.show()



# Visualisation en 3D de du data set et de la courbe sigmoïde (c'est pas moi ça)
fig = go.Figure(data=[go.Scatter3d( 
    x=X[:, 0].flatten(),
    y=X[:, 1].flatten(),
    z=y.flatten(),
    mode='markers',
    marker=dict(
        size=5,
        color=y.flatten(),                
        colorscale='YlGn',  
        opacity=0.8,
        reversescale=True
    )
)])

# Fonction sigmoïde 
fig.update_layout(template= "plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
fig.layout.scene.camera.projection.type = "orthographic"
fig.show()

X0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
X1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
xx0, xx1 = np.meshgrid(X0, X1)
Z = W[0] * xx0 + W[1] * xx1 + b
A = 1 / (1 + np.exp(-Z))

fig = (go.Figure(data=[go.Surface(z=A, x=xx0, y=xx1, colorscale='YlGn', opacity = 0.7, reversescale=True)]))

fig.add_scatter3d(x=X[:, 0].flatten(), y=X[:, 1].flatten(), z=y.flatten(), mode='markers', marker=dict(size=5, color=y.flatten(), colorscale='YlGn', opacity = 0.9, reversescale=True))


fig.update_layout(template= "plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
fig.layout.scene.camera.projection.type = "orthographic"
fig.show()
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
"""
Demo Investigacion GNNs Estructuras Discretas
Autores: Sebastián Garro Granados, Joel Brenes Vargas, Efraín Ignacio Retana Segura
Grupo: 02-10 am

Antes de nada, las bases para hacer una NN de esta manera se hizo usando el estudio/tutorial: 
https://colab.research.google.com/github/lucmos/DLAI-s2-2020-tutorials/blob/master/09/GCN.ipynb#scrollTo=Vk7Gd5DnhEGG
Hecho por Luca Moschella y Antonio Norelli
Esto se tomo solo para la base de la GCN y las layers
"""
G = nx.karate_club_graph()
n_nodes = G.number_of_nodes()


# Matriz de Adyacencia con self-loops y normalización 
A = nx.adjacency_matrix(G).toarray()
A = torch.tensor(A, dtype=torch.float)
A_hat = A + torch.eye(n_nodes)
D_hat = torch.diag(torch.sum(A_hat, dim=1))
D_hat_inv_sqrt = torch.diag(torch.pow(D_hat.diag(), -0.5))
L = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

# Features: tipo one-hot 
X = torch.eye(n_nodes)

# Etiquetas reales del dataset 
labels = torch.tensor([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in range(n_nodes)])

# Nodos de entrenamiento (3 y 3 para que este balanceado), no es necesario poner más de un nodo de entrenamiento de cada lado, sin embargo ayuda con la consistencia 
# del resultado al correr el programa varias veces 
# Mr. Hi: 0, 1, 2 ; Officer: 33, 32, 31
train_indices = torch.tensor([0, 1, 2, 31, 32, 33])
print("Etiquetas de entrenamiento:", labels[train_indices].tolist())

# Se define una capa GCN 
class GCNLayer(nn.Module):
    def __init__(self, propagator, in_features, out_features, dropout=0.3):
        super().__init__()
        self.propagator = propagator
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = self.dropout(X)
        X = self.propagator @ X
        return F.relu(self.fc(X))

# Modelo completo (de dos capas)
class GCN(nn.Module):
    def __init__(self, propagator):
        super().__init__()
        self.gcn1 = GCNLayer(propagator, n_nodes, 64)
        self.gcn2 = GCNLayer(propagator, 64, 2, dropout=0.0)

    def forward(self, X):
        X = self.gcn1(X)
        X = self.gcn2(X)
        return F.log_softmax(X, dim=1)

model = GCN(L)

# Inicialización de pesos
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Predicción antes del entrenamiento
with torch.no_grad():
    y_pred_before = torch.argmax(model(X), dim=1)

# Entrenamiento
def train(model, optimizer, X, y, train_indices, epochs=150):
    for epoch in range(epochs + 1):
        model.train()
        out = model(X)
        loss = F.nll_loss(out[train_indices], y[train_indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# Se aplica el entrenamiendo
train(model, optimizer, X, labels, train_indices)

# Predicción después del entrenamiento
with torch.no_grad():
    model.eval()
    y_pred_after = torch.argmax(model(X), dim=1)
    acc = (y_pred_after == labels).float().mean().item()

print("\nDistribución antes:", Counter(y_pred_before.tolist()))
print("Distribución después:", Counter(y_pred_after.tolist()))
print("Accuracy final: {:.2f}%".format(acc * 100))

# Visualización 
def visualize_comparison(y_pred_before, y_pred_after):
    pos = nx.spring_layout(G, seed=42)
    cmap = {0: 'red', 1: 'blue'}

    plt.figure(figsize=(18, 5))

    # Clases reales del dataset
    plt.subplot(1, 3, 1)
    nx.draw(G, pos, node_color=[cmap[i.item()] for i in labels], with_labels=True)
    plt.title("Clases reales")

    # Predicciones antes del entrenamiento
    plt.subplot(1, 3, 2)
    nx.draw(G, pos, node_color=[cmap[i.item()] for i in y_pred_before], with_labels=True)
    plt.title("Antes del entrenamiento")

    # Predicciones después del entrenamiento
    plt.subplot(1, 3, 3)
    nx.draw(G, pos, node_color=[cmap[i.item()] for i in y_pred_after], with_labels=True)
    plt.title(f"Después del entrenamiento\nAcc: {acc * 100:.2f}%")

    plt.tight_layout()
    plt.show()

visualize_comparison(y_pred_before, y_pred_after)

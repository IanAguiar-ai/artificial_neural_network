# -*- coding: utf-8 -*-
"""Example_artificial_neural_network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fQKumDvIomGGaLM-ESZM4cJ06joKhNvu

# Download the Library
"""

! pip install git+https://github.com/IanAguiar-ai/artificial_neural_network

from neural_network import artificial_neural_network as nn

help(nn)

"""# MLP classifier with this library:

## For comparison, using ready-made library:
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

"""## Creating and viewing the network:"""

rede = nn.mlp([len(X[0]), 10, 10, 3], learn = 0.05)

rede

"""## Training the model:"""

X_train = X_train.tolist()
X_test = X_test.tolist()
y_train = y_train.tolist()
y_test = y_test.tolist()

def one_hot(a):
  k = [0, 0, 0]
  k[a] = 1
  return k

rede.train(X_train, list(map(one_hot, y_train)), times = 100)

"""## Testing the results:"""

ac, total = 0, 0
for i in range(len(X_test)):
  resp = (rede == X_test[i])
  if resp.index(max(resp)) == y_test[i]:
    ac += 1
  total += 1

print(f"\n\nAcurracy {ac/total * 100}")

"""In this specific case, the library made from scratch was better than the ready-made implementation."""
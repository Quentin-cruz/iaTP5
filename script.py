import pandas as pd
import random

dataFrame1MonJeuDeDonneeDentrainement = pd.read_csv("donnesTestIa.csv")
dataFrame1MonJeuDeDonneeDentrainement["danger"] = dataFrame1MonJeuDeDonneeDentrainement["danger"].map({"safe": 0, "dangerous": 1})

w = 1.0
b = -79.5
print(w, b)

X = dataFrame1MonJeuDeDonneeDentrainement["vitesse"].tolist()
Y = dataFrame1MonJeuDeDonneeDentrainement["danger"].tolist()
print(X[:5], Y[:5])

""""def test(a, b):
    calcul = a * b
    return calcul"""
    
""""for x, y in zip(X, Y):
    y_pred = Zpredict(x)
    erreur = abs(y - y_pred)
    print(f"x = {x}, y = {y}, y_pred = {y_pred}, erreur = {erreur}")"""""

def Zpredict(x):
    Z = w * x + b
    if Z > 0:
        return 1
    else:
        return 0

eta = 0.1
epochs = 500
w = 0.0
b = 0.0

# Normalisation de X
vmax = max(X)
Xn = [x / vmax for x in X]

for epoch in range(epochs):
    erreurs = 0
    for x, y in zip(Xn, Y):
        y_pred = Zpredict(x)
        erreur = y - y_pred
        if erreur != 0:
            w += eta * erreur * x
            b += eta * erreur
            erreurs += 1
    print(f"Époque {epoch+1} : w = {w:.2f}, b = {b:.2f}, erreurs = {erreurs}")
    if erreurs == 0:
        print("Apprentissage terminé.")
        break
for x in X:
    print(f"vitesse = {x} → danger prédit = {Zpredict(x / vmax)}")
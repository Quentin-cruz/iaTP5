import pandas as pd
import random

dataFrame1MonJeuDeDonneeDentrainement = pd.read_csv("donnesTestIa.csv")
dataFrame1MonJeuDeDonneeDentrainement["danger"] = dataFrame1MonJeuDeDonneeDentrainement["danger"].map({"safe": 0, "dangerous": 1})

w = random.randint(-10, 10)
b = random.randint(-10, 10)
print(w, b)

X = dataFrame1MonJeuDeDonneeDentrainement["vitesse"].tolist()
Y = dataFrame1MonJeuDeDonneeDentrainement["danger"].tolist()
print(X[:5], Y[:5])

""""def test(a, b):
    calcul = a * b
    return calcul"""

def Zpredict(x):
    Z = w * x + b
    if Z > 0:
        return 1
    else:
        return 0

""""for x, y in zip(X, Y):
    y_pred = Zpredict(x)
    erreur = abs(y - y_pred)
    print(f"x = {x}, y = {y}, y_pred = {y_pred}, erreur = {erreur}")"""""
    

eta = 1.0
epochs = 10
for epoch in range(epochs):
    erreurs = 0
    for x, y in zip(X, Y):
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
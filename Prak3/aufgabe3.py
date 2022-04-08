'''
###################
#### Aufgabe 3 ####
###################

Prak 2 betrachten

Es soll beliebig viele Inputs geben n.

y soll eine Summe aus einer Teilmenge m von n sein. m <= n

'''
import random
from tkinter import N

from sqlalchemy import false

# Initialisierung
inputCount = 10
nEpochs = 6000 # 1000 viel zu klein, 2000 knap zu klein, 3000 fast perfekt; lr=0.001
lr = 0.001

n = 5
m = 3


inputs = [[random.randint(-10,10) for i in range(n)] for i in range(inputCount)] 
weights = [2,3,4,5,6]

## Ungenutzt
def loss_function(y, y_caret):
    return 1/2 * (y_caret-y)^2

## (^y-y)*z
def partInt(y_dach, y, x):
    return (y_dach - y)*x

if __name__ == '__main__':
    print(f"Start Weights: {weights}")
    print("##############")
    for j in range(0, nEpochs):
        print("#################")
        print("## Neue Epoche ##")
        print("#################")
        for input in inputs:
            print(f"Input: {input}; m = {m}")
            # Wunschziel, Ergebnis und Abweichung berechnen
            y = 0
            for i in range(m):
                y += input[i]

            y_caret = 0
            for i in range(n):
               y_caret = input[i] * weights[i]
               
            d = y_caret-y

            if(abs(d) > 0):
                # Gewichte anpassen
                for i in range(n):
                    partial_w = partInt(y_caret,y,input[i])
                    weights[i] -= (lr*partial_w)

            ## Output only
            y = 0
            for i in range(m):
                y += input[i]

            y_caret = 0
            for i in range(n):
               y_caret = input[i] * weights[i]

            d = y_caret-y

            print(f"Weights:{weights}")
            print(f"Current: {y_caret}; Target: {y}")
            print(f"Abweichung y-^y: {d}")
            print("##############")
            
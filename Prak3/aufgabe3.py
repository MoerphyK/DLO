'''
###################
#### Aufgabe 3 ####
###################

Prak 2 betrachten

Es soll beliebig viele Inputs geben n.

y soll eine Summe aus einer Teilmenge m von n sein. m <= n

'''
import random
import numpy as np

# Initialisierung
inputCount = 10 # Anzahl Inputs
nEpochs = 20000
batchSize = 2
lr = 0.0001
n = 6 # Dimension Inputs !! max 10
m = 3

weights = [3,2,3,4,1,2,3,4,5,0]
weights = weights[:n]
inputs = [[random.randint(-5, 5) for i in range(n)] for i in range(inputCount)]

## Ungenutzt
def loss_function(y, y_caret):
    return 1/2 * (y_caret-y)^2

## (^y-y)*z
def partInt(y_dach, y, x):
    return (y_dach - y)*x

def averageList(list,i):
    sum = 0
    for e in list:
        sum += e[i]
    return sum/len(list)

if __name__ == '__main__':
    print(f"Start Weights: {weights}")
    print("##############")
    for j in range(0, nEpochs):
        # print("#################")
        # print("## Neue Epoche ##")
        # print("#################")

        random.shuffle(inputs)
        batches = np.array_split(inputs,inputCount/batchSize)

        for batch in batches:
            modifications = []
            for input in batch:
                #print(f"Input: {input}; m = {m}")
                # Wunschziel, Ergebnis und Abweichung berechnen
                y = 0
                for i in range(m):
                    y += input[i]

                y_caret = 0
                for i in range(n):
                   y_caret += input[i] * weights[i]

                d = y_caret-y

                # Gewichte anpassen

                mod_list = []
                for i in range(n):
                    partial_w = partInt(y_caret,y,input[i])
                    mod_list.append(weights[i] - (lr*partial_w))
                modifications.append(mod_list)

            for i in range(n):
                weights[i] = averageList(modifications,i)


    for input in inputs:
        y = 0
        y_all = 0
        y_caret = 0
        for i in range(len(input)):
            if i < m:
                y += input[i]
            y_all += input[i]
            y_caret += input[i] * weights[i]

        print(f"Weights:{weights}")
        print(f"Input:{input}")
        print(f"Current: {y_caret}; Target m = {m}: {y}; Target all: {y_all}")
        print("##############")
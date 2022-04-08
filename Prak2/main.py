'''
###################
#### Aufgabe 2 ####
###################

1. Implementieren Sie das einfache Netz von Folie 69
Tipps:
Berechnet: ^y = x1*w1 + x2*w2
Gewünschte Ausgabe: y = x1+x2
Fehlerfunktion (loss function): L(^y,y) = (1/2)*(^y-y)²

2. Implementieren Sie auch das zugeh¨orige Training mittels Gradientenabstieg (Folie 73), gerne mit Mini-Batches

3. Nutzen Sie zuf¨allig erzeugte Beispiele zum Trainieren (y = x1 + x2)

4. Fur welche Werte von ¨ η funktioniert das Training besser / schlechter / nicht?

'''
import random

class Point:
    def __init__(self):
        self.x = random.randint(-100,100)
        self.y = random.randint(-100,100)

# Initialisierung
w1 = -1
w2 = -2
nEpochs = 10
lr = 0.01

## Ungenutzt
def loss_function(y, y_caret):
    return 1/2 * (y_caret-y)^2

## (^y-y)*z
def partInt(y_dach, y, x):
    return (y_dach - y)*x

if __name__ == '__main__':
    #Eingabewerte erstellen
    input = []  # input
    p = Point()
    p.x = 1
    p.y = 1
    input = [p]

    ''' Mehrere Zufallspunkte erzeugen
    for i in range(0, 10):  # Quadrant 1
        p = Point()
        input.append(p)
    '''

    print(f"Start Input: {input[0].x},{input[0].y}")
    print(f"Start Weights: {w1}; {w2}")
    print("##############")
    for j in range(0, nEpochs):
        for i in range(0, len(input)):
            # Wunschziel, Ergebnis und Abweichung berechnen
            y = int(input[i].x) + int(input[i].y)
            y_caret = input[i].x * w1 + input[i].y * w2
            d = y_caret-y

            print(f"Abweichung y-^y: {d}")

            ## Zweiter Lösungsansatz
            # Part Ableitung nach W1 -> Falls positiv w1 verkleinern
            partial_w1 = partInt(y_caret, y, input[i].x)
            if (partial_w1 > 0):
                # W1 anpassen
                w1 -= (lr * partial_w1)
            else:
                w1 += (lr * partial_w1)

            # Part Ableitung nach W2 -> Falls positiv w2 verkleinern
            partial_w2 = partInt(y_caret, y, input[i].y)
            if (partial_w2 > 0):
                # W2 anpassen
                w2 -= (lr * partial_w2)
            else:
                w2 += (lr * partial_w2)

            # Ausgabe
            print(f"Weights:{w1},{w2} -> {input[i].x * w1 + input[i].y * w2}")
            print(f"Aktuelles Ergebnis: {input[0].x * w1},{input[0].y * w2}")
            print("##############")

            '''
            ## Erster Lösungsansatz:
            if(abs(d) > 0):
                # W1 anpassen
                partial_w1 = partInt(y_caret,y,input[i].x)
                w1 -= (lr * partial_w1)

                # W2 anpassen
                partial_w2 = partInt(y_caret,y, input[i].y)
                w2 -= (lr * partial_w2)
            '''

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
        self.x = random.randint(-10,10)
        self.y = random.randint(-10,10)

# Initialisierung
w1 = -1
w2 = -2
nEpochs = 3000 # 1000 viel zu klein, 2000 knap zu klein, 3000 fast perfekt; lr=0.001
lr = 0.001

## Ungenutzt
def loss_function(y, y_caret):
    return 1/2 * (y_caret-y)^2

## (^y-y)*z
def partInt(y_dach, y, x):
    return (y_dach - y)*x

if __name__ == '__main__':
    #Eingabewerte erstellen
    inputs = []  # input
    
    ## Mehrere Zufallspunkte erzeugen
    for i in range(0, 10):  # Quadrant 1
        p = Point()
        inputs.append(p)
    
    ## Manuell einen Punkt anlegen
    # p = Point()
    # p.x = 1
    # p.y = 1
    # input = [p]

    # print(f"Start Input: {input[0].x},{input[0].y}; Target = {input[0].y + input[0].x}")
    print(f"Start Weights: {w1}, {w2}; Ziel 1 & 1")
    print("##############")
    for j in range(0, nEpochs):
        for input in inputs:
            print(f"Input: {input.x} & {input.y}")
            # Wunschziel, Ergebnis und Abweichung berechnen
            y = int(input.x) + int(input.y)
            y_caret = input.x * w1 + input.y * w2
            d = y_caret-y

            print(f"Abweichung y-^y: {d}")

            '''
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

            '''
            
            ## Erster Lösungsansatz:
            if(abs(d) > 0):
                # W1 anpassen
                partial_w1 = partInt(y_caret,y,input.x)
                w1 -= (lr * partial_w1)

                # W2 anpassen
                partial_w2 = partInt(y_caret,y, input.y)
                w2 -= (lr * partial_w2)
            
            # Ausgabe
            print(f"Weights:{w1},{w2} -> Ergebnis {input.x * w1 + input.y * w2}")
            print(f"Target: {input.x + input.y}")
            print("##############")
            
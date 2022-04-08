import sys
from turtle import tracer

weights = [2,3,2,5]
target = 1
n = 0.1

## Check Inputs and compare to weights length
def checkInput(length):
    input = []
    try:
        if len(sys.argv)-1 == length:
            return list(map(int, sys.argv[1:]))
        else:
            raise BaseException("Please pass exactly "+str(length)+" Integer arguments")    
    except:
        raise

def checkXW(inputs,weights):
    counter = 0
    sum = 0
    while counter < len(inputs):
        sum += inputs[counter]*weights[counter]
        counter +=1
    if sum <= target:
        return -1
    else:
        return 1
    

def checkError(y_d):
   return target - y_d 


if __name__ == '__main__':
    inputs = checkInput(len(weights))
    print(f"Input: {inputs}, Weights: {weights}, Target: {target}, Learningrate:{n}")
    d = 1
    t = 1
    while (d != 0):
        y_d = checkXW(inputs,weights)
        d = checkError(y_d)
        print()
        print("###########")
        print(f"Run t={t}")
        print(f"^y={y_d}")
        print(f"Weights:{weights}")
        print(f"d={d}")
        if d == 0:
            print(f"Success with weights: {weights} in run t: {t}")
            break
        else:
            counter = 0
            while counter < len(inputs):
                weights[counter] += inputs[counter]*d*n
                counter +=1

        t +=1
    print()
    print("#########################")
    print("Proof:")
    sum = 0
    counter = 0
    while counter < len(inputs):
        sum += weights[counter]*inputs[counter]
        print(f"{weights[counter]} * {inputs[counter]} = {weights[counter]*inputs[counter]}")
        counter +=1
    print(f"(u) {sum} > {target} (Theta) -> {target} & Target")
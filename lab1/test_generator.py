import random
import os


def generate():
    MAX_SIZE = 2 ** 25
    PRECISION = 10

    size = MAX_SIZE - 1

    vec = [round(random.uniform(-MAX_SIZE, MAX_SIZE), random.randint(0, PRECISION)) for _ in range(size)]

    dirName = r'B:\VisualStudio\source\lab1\lab1'
    path = os.path.join(dirName, 'array')
    with open(dirName, 'w') as f:
        for i in range(len(vec)):
            if i != len(vec) - 1:
                f.write(str(vec[i]) + '\n')
            else:
                f.write(str(vec[i]))
                
    vec.reverse()
    path = os.path.join(dirName, 'answer')
    with open(dirName, 'w') as f:
    for i in range(len(vec)):
        if i != len(vec) - 1:
            f.write(str(vec[i]) + '\n')
        else:
            f.write(str(vec[i]))
    # =======================

if __name__ == "__main__":
    generate()

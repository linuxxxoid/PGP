import random
import os


def generate():
    MAX_SIZE = 2 ** 25
    PRECISION = 10

    size = 100
    vec = []
    for i in range(size):
    	vec.append(round(random.uniform(0, 555.0), random.randint(0, PRECISION)))

    dirName = '/Users/linuxoid/Desktop/VUZICH/PGP'
    path = os.path.join(dirName, 'array')
    print(path)
    with open(dirName, 'w') as f:
        f.write(str(len(vec)) + '\n')
        for i in range(len(vec)):
            if i != len(vec) - 1:
                f.write(str(vec[i]) + '\n')
            else:
                f.write(str(vec[i]))

    vec.reverse()
    path = os.path.join(dirName, 'answer')
    with open(dirName, 'w') as f:
        f.write(str(len(vec)) + '\n')
        for i in range(len(vec)):
            if i != len(vec) - 1:
                f.write(str(vec[i]) + '\n')
            else:
                f.write(str(vec[i]))


if __name__ == "__main__":
    generate()
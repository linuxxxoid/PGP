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
        f.write(vec)
    
    vec.reverse()
    path = os.path.join(dirName, 'answer')
    with open(path, 'w') as f:
            f.write(vec)
    # =======================

if __name__ == "__main__":
    generate()
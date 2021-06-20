import os
import random


if __name__ == "__main__":
	path = './small'
	size = 5
	with open(path, 'w') as f:
		f.write(str(size) + '\n')	
		for i in range(size + 1):
			for j in range(size):
				f.write(str(random.randint(1, 100)) + ' ')
			f.write('\n')

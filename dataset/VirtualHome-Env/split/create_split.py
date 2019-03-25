import numpy as np
from glob import glob


np.random.seed(123)

def _func(p):
    for i in range(1, 8):
        p = p.replace('TrimmedTestScene{}_graph/'.format(i), '')
    return p

prog_paths = glob('../original_programs/executable_programs/*/*/*txt')
prog_paths = list(set([_func(i) for i in prog_paths]))
prog_paths = np.sort(np.array(prog_paths))

np.random.shuffle(prog_paths)

n = len(prog_paths)

f = open('train_progs_path.txt', 'w')
for i in prog_paths[:int(0.7*n)]:
    i = '/'.join(i.split('/')[3:]).replace('.txt', '')
    f.write(i)
    f.write('\n')

f = open('test_progs_path.txt', 'w')
for i in prog_paths[int(0.7*n):]:
    i = '/'.join(i.split('/')[3:]).replace('.txt', '')
    f.write(i)
    f.write('\n')


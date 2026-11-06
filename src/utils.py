import numpy as np
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
def stamp(name, M):
    print(f"{name} =")
    print(np.atleast_2d(M))

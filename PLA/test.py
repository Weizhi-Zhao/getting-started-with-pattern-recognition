import numpy as np
import matplotlib.pyplot as plt

class test:
    def __init__(self, a):
        self.a = a

    def p(self, x):
        print(self.a, x)

def pp(pfunction, x):
    print(pfunction)
    pfunction(x)

x = test(1)
y = test(2)
pp(x.p, 10)
pp(y.p, 20)
pp(test.p, 30)
print(x.p, y.p, test.p)
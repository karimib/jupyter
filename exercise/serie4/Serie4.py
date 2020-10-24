import numpy as np
import matplotlib.pyplot as plt
# Das Polynom f(x) besitzt zwei reelle Nullstellen x_1 im intervall [-1,0] und x_2 im Intervall [0,1]
x=np.linspace(-1, 1, 100, endpoint=True)
y=230*x**4 + 18*x**3 + 9*x**2 - 221*x - 9
z=np.zeros(np.size(x))
plt.plot(x,z,'b-', label='x-Achse')
plt.plot(x,y,'r-', label='y=p(x)')
plt.legend(loc='lower left')
plt.title('Funktionsgraph')
plt.xlabel('x-Achse')
plt.ylabel('y-Achse')
plt.show()

def f(x):
    y = (230*x**4 + 18*x**3 + 9*x**2 - 221*x - 9)
    return y


def F(x):
    y = (230*x**4 + 18*x**3 + 9*x**2 - 9) / 221
    return y

x_approx=np.zeros(iter)
x_approx[0] = g(x)
x[0] = -1.0

for k in range(noIterations-1):
    x_approx[k+1]=g(x_approx[k])
    s = 'Iteration: '+repr(k+1)+', x-value: '+repr(x[k+1])
    print(s)

f=g(x)
plt.plot(x_approx,f,'r-', label='y=f(x)')
plt.plot(x_approx,x_approx,'b-', label='y=x')
plt.plot(x_approx[0:noIterations-1],x_approx[1:noIterations],'g*', label='x+1=f(x)')
plt.legend(loc='upper left')
plt.title('Fixpunkt-Iteration')
plt.xlabel('x-Achse')
plt.ylabel('y-Achse')
plt.show()


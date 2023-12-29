import numpy as np
import matplotlib.pyplot as plt

N_POINTS = 10
w_1, w_0 = 1.2, 2

x = np.random.normal(scale=1, loc=1, size=(N_POINTS,))
x = x - np.min(x)

# create a linear relationship and add some noise/error
y = x*w_1 + w_0 + np.random.normal(scale=0.1, size=x.shape)

x_linspace = np.linspace(np.min(x), np.max(x), 5)
y_linspace = x_linspace*w_1 + w_0
plt.scatter(x, y, color='red') # draw the points
plt.plot(x_linspace, y_linspace, color='black') # draw the points
plt.xlabel('x')
plt.ylabel('y')
plt.show()
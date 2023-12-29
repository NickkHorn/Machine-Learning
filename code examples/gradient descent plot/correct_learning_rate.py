import numpy as np
import matplotlib.pyplot as plt

lr = 0.2
x0, y0 = -2, 4
x = [x0]
y = [y0]


for i in range(15):
    x0 -= lr*2*x0
    y0 = x0**2

    plt.annotate("", xytext=(x[-1], y[-1]), xy=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="red", lw=3))
    x.append(x0)
    y.append(y0)

xs = np.linspace(-2.2,2.2, 100)
plt.plot(xs, xs**2, color='black')
plt.show()
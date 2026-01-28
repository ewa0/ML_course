import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 100)
y = x * 2
z = x ** 2

fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(x, y, 'b', '-')
axes[1].plot(x, z, 'r')
plt.tight_layout()
plt.show()

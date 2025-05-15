import matplotlib.pyplot as plt
import pandas
import numpy as np

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [i**2 for i in x]
plt.title("$y=x^2$")
plt.plot(x, y)
plt.show()
print(np.zeros(9))

import numpy as np
a = np.arange(5)
b = np.array([0,0,0,0,1])
b.put(a,True)
print(b)

import numpy as np

x = np.array([[0, 0, 0, 0],
              [0, 255, 255, 0],
              [0, 255, 255, 0],
              [0, 0, 0, 0]])
where = np.where(x)
where2 = np.vstack((where[0], where[1]))
print(where)
print(where2)
y1, x1 = np.amin(where2, axis=1)
print(y1, x1)
y2, x2 = np.amax(where2, axis=1)
print(y2, x2)

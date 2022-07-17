import numpy as np
from sklearn.linear_model import LinearRegression

a = np.array([[1, 1, 1], [2, 3, 4], [5, 4, 3], [8, 7, 4], [32, 31, 22]])
w = np.array([3, 2, 3])
y = np.matmul(w, a.T)

print(a)
print(y)

lr = LinearRegression()
lr.fit(a, y)

print(lr.coef_)
# # array([0.2, 0.5])
# 
# print(lr.intercept_)
# # 0.099
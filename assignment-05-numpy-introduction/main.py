import numpy as np

arr = np.array([10, 20, 30, 40, 50])
print('Q1: Add 2', arr + 2)
print('Multiply by 3', arr * 3)
print('Divide by 2', arr / 2)

arr = np.array([1, 2, 3, 6, 4, 5])
print('Q2a: Reverse', arr[::-1])
for name, values in [('x', np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])), ('y', np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3]))]:
    unique, counts = np.unique(values, return_counts=True)
    modes = unique[counts == counts.max()]
    print('Q2b:', name, 'Most frequent', modes, 'Indices', {int(v): np.where(values == v)[0].tolist() for v in modes})

matrix = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print('Q3: First row second column', matrix[0, 1])
print('Third row first column', matrix[2, 0])

Harshveer_Singh = np.linspace(10, 100, 25)
print('Q4: Array', Harshveer_Singh)
print('Dimensions', Harshveer_Singh.ndim, 'Shape', Harshveer_Singh.shape, 'Size', Harshveer_Singh.size, 'Dtype', Harshveer_Singh.dtype, 'Bytes', Harshveer_Singh.nbytes)
print('Reshape as column vector', Harshveer_Singh.reshape(-1, 1))
print('Transpose of column vector', Harshveer_Singh.reshape(1, -1).T)
print('T of 1D array keeps shape', Harshveer_Singh.T.shape)

ucs420_harshveer = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 15, 20, 35]])
print('Q5: Mean', ucs420_harshveer.mean(), 'Median', np.median(ucs420_harshveer), 'Max', ucs420_harshveer.max(), 'Min', ucs420_harshveer.min(), 'Unique', np.unique(ucs420_harshveer))
reshaped_ucs420_harshveer = ucs420_harshveer.reshape(4, 3)
resized_ucs420_harshveer = np.resize(ucs420_harshveer, (2, 3))
print('Reshaped', reshaped_ucs420_harshveer)
print('Resized', resized_ucs420_harshveer)

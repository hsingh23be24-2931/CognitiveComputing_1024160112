import numpy as np

temperature = np.array([25, 28, 31, 35, 38, 27, 33, 40])
corrected = temperature + 2
print('Q1: Corrected Celsius', corrected)
print('Fahrenheit of corrected readings', corrected * 9 / 5 + 32)
print('Corrected readings above 32 C', corrected[corrected > 32])
print('Number above 32 C', np.count_nonzero(corrected > 32))
print('Vectorization applies operations to complete arrays; Boolean indexing filters matching observations without Python loops.')

steps = np.array([[5000, 6200, 7100], [8000, 7500, 9000], [4500, 5100, 4800], [9000, 8500, 9500]])
print('Q2: Total', steps.sum(), 'Mean', steps.mean(), 'Max', steps.max(), 'Min', steps.min())
print('Total by day', steps.sum(axis=0), 'Total by user', steps.sum(axis=1))
print('Maximum position (user, day), zero-indexed', np.unravel_index(steps.argmax(), steps.shape))

original = np.array([1, 2, 3, 4, 5, 6])
subset = original[1:5]
subset[0] = 999
print('Q3: Original after view change', original, 'Subset', subset)
copied = original[1:5].copy()
copied[0] = 500
print('Original after copy change', original, 'Copied', copied)
matrix = np.arange(1, 13).reshape(3, 4)
print('Matrix', matrix)
print('First row', matrix[0], 'Last row', matrix[-1])
print('Second column', matrix[:, 1], 'Rows 1-2 columns 2-3', matrix[0:2, 1:3])
flat = matrix.flatten()
raveled = matrix.ravel()
raveled[0] = 200
print('After ravel modification', matrix, 'Ravel', raveled)
flat[0] = 500
print('After flatten modification', matrix, 'Flatten', flat)
print('Shape', matrix.shape, 'ndim', matrix.ndim, 'size', matrix.size, 'dtype', matrix.dtype)

X = np.array([[6, 70, 3], [5, 50, 6], [8, 80, 2], [4, 30, 8]], dtype=float)
y = np.array([40, 65, 30, 85], dtype=float)
print('Q4: X shape', X.shape, 'Dimensions', X.ndim)
print('Transpose', X.T)
print('The transpose exchanges sample rows and feature columns.')
xtx = X.T @ X
print('X.T @ X', xtx)
print('Inverse', np.linalg.inv(xtx))
beta = np.linalg.inv(xtx) @ X.T @ y
print('OLS coefficients', beta)
print('Each coefficient represents the change in predicted score for a one-unit increase of its feature, holding the others fixed.')
new_user = np.array([5, 40, 7])
print('Predicted assistance score', new_user @ beta)

import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        ans = np.zeros(len(z))
        for i in range(len(z)):
            new_z = 1/(1+np.exp(-z[i]))
            ans[i] = new_z
        return np.round(ans, 5)


    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        ans = np.zeros(len(z))
        for i in range(len(z)):
            ans[i] = max(0,z[i])
        return ans

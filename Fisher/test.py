from Fisher import Fisher
import numpy as np

c1 = np.array(
        [
            [1, -1],
            [-1, 4]
        ]
    )

c2 = np.array([2, 3])

print(c2)
print(c2[:, None])
print(c2[None, :])

print(c2[:, None] @ c2[None, :], c2 @ c1)
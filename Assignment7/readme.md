## Receptive Field Calculation Assignment 7A

![](https://devblogs.nvidia.com/wp-content/uploads/2015/08/image6.png)


| Layer #            | Kernel Size | Stride | Receptive Field | Jump |
| ------------------ | ----------- | ------ | --------------- | ---- |
| Conv(7x7) S=2      | 7           | 2      | 7x7             | 1    |
| MaxPool(3x3) S=2   | 3           | 2      | 11x11           | 2    |
| Conv(3x3) S=1      | 3           | 1      | 19x19           | 4    |
| MaxPool(3x3) S=2   | 3           | 2      | 27x27           | 4    |
| 3(a) Conv(5x5) S=1 | 5           | 1      | 59x59           | 8    |
| 3(b) Conv(5x5) S=1 | 5           | 1      | 91x91           | 8    |
| MaxPool(3x3) S=2   | 3           | 2      | 107x107         | 8    |
| 4(a) Conv(5x5) S=1 | 5           | 1      | 171x171         | 16   |
| 4(b) Conv(5x5) S=1 | 5           | 1      | 235x235         | 16   |
| 4(c) Conv(5x5) S=1 | 5           | 1      | 299x299         | 16   |
| 4(d) Conv(5x5) S=1 | 5           | 1      | 363x363         | 16   |
| 4(e) Conv(5x5) S=1 | 5           | 1      | 427x427         | 16   |
| MaxPool(3x3) S=2   | 3           | 2      | 459x459         | 16   |
| 5(a) Conv(5x5) S=1 | 5           | 1      | 587x587         | 32   |
| 5(b) Conv(5x5) S=1 | 5           | 1      | 715x715         | 32   |
| Conv(7x7) S=2      | 7           | 2      | 907x907         | 32   |

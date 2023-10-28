import numpy as np

class Softmax:
    def __init__(self, 
                 x: np.ndarray, 
                 y:np.ndarray
                ) -> None:
        # y: one-hot
        self.n_class = y.shape[1]
        self.x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        self.y = y
        self.w = np.random.normal(0, 0.01, (self.n_class, self.x.shape[1]))

    def train(self, epochs, lr, batch_size=1):
        loss_list = []
        acc_list = []
        n = self.x.shape[0]
        for _ in range(epochs):
            data = np.concatenate((self.x, self.y), axis=1)
            np.random.shuffle(data)
            for i in range(0, n, batch_size):
                batch = data[i:min(i + batch_size, n), :]
                batch_x = batch[:, :-self.n_class]
                batch_y = batch[:, -self.n_class:]
                y_hat = self(batch_x, augment=False)
                grad = (y_hat - batch_y).T @ batch_x / batch_x.shape[0]
                self.w -= lr * grad
            y_hat = self(self.x, augment=False)
            loss = -np.log((y_hat * self.y).sum(axis=1)).sum() / n
            loss_list.append(loss)
            acc = np.equal(np.argmax(y_hat, axis=1), np.argmax(self.y, axis=1)).sum() / n
            acc_list.append(acc)
        return loss_list, acc_list

    def __call__(self, x: np.ndarray, augment=True):
        if augment:
            x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        x = np.exp(x @ self.w.T)
        x = x / x.sum(axis=1, keepdims=True)
        return x
    
if __name__ == "__main__":
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[1, 0], [0, 1], [1, 0]])
    s = Softmax(x, y)
    s.train(1, 1)
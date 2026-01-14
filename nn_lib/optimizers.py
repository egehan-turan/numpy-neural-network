import numpy as np

class SGD:
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {} 

    def optimize(self, layers):
        for layer in layers:

            for param in layer.get_parameters():

                # lazily initilize velocities
                if param not in self.velocity:
                    self.velocity[param] = np.zeros_like(param.data)

                v = self.momentum * self.velocity[param] + param.grad
                self.velocity[param] = v
                
                param.data -= self.learning_rate * v

class Adam:
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.m = {}
        self.v = {} 
        self.t = 0


    def optimize(self, layers):
        self.t += 1
        for layer in layers:

            for param in layer.get_parameters():

                # lazily initilize momentum and variance
                if param not in self.m:
                    self.m[param] = np.zeros_like(param.data)
                    self.v[param] = np.zeros_like(param.data)

                self.m[param] = self.beta_1 * self.m[param] + (1 - self.beta_1) * param.grad
                self.v[param] = self.beta_2 * self.v[param] + (1 - self.beta_2) * np.square(param.grad)

                m_normalized = self.m[param] / (1 - self.beta_1 ** self.t)
                v_normalized = self.v[param] / (1 - self.beta_2 ** self.t)
                
                param.data -= self.learning_rate * m_normalized / (np.sqrt(v_normalized) + self.epsilon)
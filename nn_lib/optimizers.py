import numpy as np

class SGD:
    def __init__(self, learning_rate: float = 0.05, momentum: float = 0.9):
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
import numpy as np

np.random.seed(0)

SCALER = 0.10

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = SCALER * np.random.randn(input_dim, output_dim)
        self.biases = np.zeros((1, output_dim))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def __str__(self) -> str:
        return f"Input dim: {self.input_dim}, output dim: {self.output_dim}"


class NeuralNetwork:
    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = None,
        output_dim: int = None,
        num_hidden_layers: int = 1,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self._init_layers(num_hidden_layers)

        self.output = None
        

    def _init_layers(self, num_hidden_layers):
        self.input_layer = Layer(self.input_dim, self.hidden_dim)
        self.layers = [self.input_layer]

        for _ in range(num_hidden_layers):
            self.layers.append(Layer(self.hidden_dim, self.hidden_dim))
        
        self.output_layer = Layer(self.hidden_dim, self.output_dim)
        self.layers.append(self.output_layer)
        

    def forward(self):
        inputs = X
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output

        return self.output_layer.output
            
    
    
nn = NeuralNetwork(input_dim=4, hidden_dim=6, output_dim=1, num_hidden_layers=2)
print(nn.forward())

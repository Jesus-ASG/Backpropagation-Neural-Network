# Import Neural Network
import Neural_Network as Nn

# Explanation:
# In this example let's give values of height and weight
# and clasiffy them in 3 categories
# 100 is under weight
# 010 is normal weight
# 001 is over weight

# Make data input and output
data_in = [
    [150, 50],
    [160, 40],
    [170, 80],
    [170, 50],
    [186, 90],
    [180, 90],
    [166, 60],
    [154, 50],
    [166, 50],
    [172, 70.5],
    [171, 86],
    [173, 88.5]
]

data_out = [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
]

# Make an object of Neural Network
network = Nn.Network()

# Optional - Modify parameters
network.error_rate = 0.01

# Assign data to Neural Network
network.set_data(data_in, data_out)

# Set a model of layers, remember last layer must be equals to size of a row output
network.make_layers([8, 10, 3])

# Train with a maximum iterations
print('Training model...')
network.train(1000)
print('Model trained!')

# Optional - show numer of iterations
print(f'Iterations: {network.it_n}')

# Start to predict
print('Ask 150, 50')
network.predict([150, 50], True)

print('Ask 170, 47')
network.predict([170, 47], True)

print('Ask 172, 72')
network.predict([172, 72], True)

print('Ask 180, 92')
network.predict([180, 92], True)

# For show all neurons values just use print(network)

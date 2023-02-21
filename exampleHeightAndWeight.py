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
network = Nn.Network(data=(data_in, data_out), model=[2, 4], activation='STEP')
network.error_rate = 0.001
network.maximum_restarts = 10

# Train with a maximum iterations
print('Training model...')
network.train(500)
print('Model trained!')

# Optional - show numer of iterations
print(f'Iterations: {network.it_n}')

# Start to predict
print('Ask 150, 50 - [0, 1, 0]')
pred = network.predict([150, 50])
print(pred)


print('Ask 170, 47 - [1, 0, 0]')
pred = network.predict([170, 47])
print(pred)


print('Ask 172, 72 - [0, 1, 0]')
pred = network.predict([172, 72])
print(pred)


print('Ask 180, 92 - [0, 0, 1]')
pred = network.predict([180, 92])
print(pred)

#print(network)
# For show all neurons values just use print(network)

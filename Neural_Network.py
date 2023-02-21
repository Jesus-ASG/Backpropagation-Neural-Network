import random
import math


class Neuron:
    def __init__(self, input_n, name):
        self.input_n = input_n
        self.name = name
        self.error = 0
        self.output = 0
        self.w0 = random.randint(-1, 1)
        self.w = list()
        for _ in range(self.input_n):
            self.w.append(random.randint(-1, 1))

    def reset(self):
        self.error = 0
        self.output = 0
        self.w0 = random.randint(-1, 1)
        self.w = list()
        for i in range(self.input_n):
            self.w.append(random.randint(-1, 1))

    def __str__(self):
        aux = f'-------- Neuron {self.name} --------\n'
        aux += f'Error / Delta = {self.error}\nOutput = {self.output}\n'
        aux += f'Inputs:\nW0 = {self.w0}\nWx = {self.w}\n'
        aux += f'-'*20
        return aux+'\n'


class Network:
    def __init__(self, data, model, activation, output_activation=''):
        self.__ERROR = False
        self.error_rate = 0.001
        self.maximum_restarts = 1
        self.current_restart = 0

        self.network = list()
        self.row_errors = list()

        self.it_n = 0

        self.data_in = data[0]
        self.data_out = data[1]

        self.set_model(model)
        
        self.f = self.gelu
        self.f_output = self.no_output_function

        match activation:
            case 'STEP':
                self.f = self.step
            case 'SIGMOID':
                self.f = self.sigmoid
            case 'RELU':
                self.f = self.relu
            case 'GELU':
                self.f = self.gelu
            case 'MISH':
                self.f = self.mish

        match output_activation:
            case 'SOFTMAX':
                self.f_output = self.softmax
            case 'SIGMOID':
                self.f_output = self.output_sigmoid


    def set_model(self, model):
        model.append(len(self.data_out[0]))
        current_neuron = 1
        for i, m in enumerate(model):
            layer = list()
            for j in range(m):
                if i == 0:  # First layer, inputs = total data
                    layer.append(Neuron(len(self.data_in[0]), str(current_neuron)))
                else:  # Other layers, number of inputs = total neurons of a layer before
                    layer.append(Neuron(model[i - 1], str(current_neuron)))
                current_neuron += 1
            self.network.append(layer)

    def train(self, max_it, it_before_restart=None):
        if self.__ERROR:
            return
        if it_before_restart is None:
            it_before_restart = int(max_it*0.95)

        for i in range(len(self.data_in)):  # Set all row error to 1
            self.row_errors.append(1)
        self.it_n = 0   # Set iteration number to 0
        b = True
        self.current_restart = 0
        while self.it_n < max_it and b:
            for rn, row in enumerate(self.data_in):  # Iterate each row of data input
                # STEP 1: Set outputs for all neurons
                for ln, layer in enumerate(self.network):
                    if ln == 0:  # Layer 0
                        for neuron in layer:
                            neuron.output = self.f(self.multiply(row, neuron.w) + neuron.w0)
                    else:   # Other layers
                        for n_n, neuron in enumerate(layer):
                            v_o = self.outputs_of_layer(self.network[ln - 1])
                            neuron.output = self.f(self.multiply(v_o, neuron.w) + neuron.w0)

                # STEP 2: Get error/delta for all neurons
                for ln, layer in reversed(list(enumerate(self.network))):  # Starting by last layer
                    if ln == len(self.network) - 1:  # Last layer
                        aux = 0
                        for n, dato in zip(layer, self.data_out[rn]):
                            n.error = dato - n.output   # Error/delta = desired output - output
                            aux += abs(n.error)   # Abs to discard -1+1 = 0, we want 0+0+...+0 = 0
                        self.row_errors[rn] = aux

                    else:  # Other layers
                        for nn, neuron in enumerate(layer):
                            error = 0
                            layer_next = self.network[ln + 1]  # Layer next to current layer
                            for n_sig in layer_next:
                                error += n_sig.error * n_sig.w[nn]
                            neuron.error = error

                # STEP 3: Update weights
                for ln, layer in enumerate(self.network):
                    for neuron in layer:
                        neuron.w0 += self.error_rate * neuron.error
                        for i, w in enumerate(neuron.w):  # Iterate weights for current neuron
                            if ln == 0:     # Layer 0
                                neuron.w[i] += self.error_rate * neuron.error * self.data_in[rn][i]
                            else:           # Other layers
                                v_o = self.outputs_of_layer(self.network[ln - 1])
                                neuron.w[i] += self.error_rate * neuron.error * v_o[i]

                # STEP 4: Check if it can end
                if self.can_finish(self.row_errors):
                    b = False
                    break

            

            # Consideration of reset when reaching n-iterations
            if self.it_n == it_before_restart and self.current_restart < self.maximum_restarts:
                for c in self.network:
                    for n in c:
                        n.reset()
                self.current_restart += 1
                self.it_n = 0
                for i in range(len(self.row_errors)):
                    self.row_errors[i] = 1
            self.it_n += 1

        
        print(f'final precision {round_array(self.row_errors, 2)}')
        
        # Add function
        # get outputs and modify values with values function
        #last_layer_outputs = self.outputs_of_layer(self.network[-1])
        #last_layer_outputs = self.f_output(last_layer_outputs)
        #for i, o in enumerate(last_layer_outputs):
        #    self.network[-1][i].output = o

    def predict(self, row):
        for ln, layer in enumerate(self.network):
            if ln == 0:  # Layer 0
                for neuron in layer:
                    neuron.output = self.f(self.multiply(row, neuron.w) + neuron.w0)
            else:  # Other layers
                for n_n, neuron in enumerate(layer):
                    v_o = self.outputs_of_layer(self.network[ln - 1])
                    neuron.output = self.f(self.multiply(v_o, neuron.w) + neuron.w0)

        last_layer = self.network[-1]   # Last layer is the layer of answers
        predictions = list()
        for layer in last_layer:
            predictions.append(layer.output)
        return round_array(self.f_output(predictions), 3)

    @staticmethod
    def outputs_of_layer(layer):    # Return array with all outputs of a layer
        aux = list()
        for x in layer:
            aux.append(x.output)
        return aux

    @staticmethod
    def multiply(valor1, valor2):  # Multiply and add valor1[i] * valor2[i]
        mul = 0
        for i in range(len(valor2)):
            mul += valor1[i] * valor2[i]
        return mul


    @staticmethod
    def step(x):
        return 0 if x < 0 else 1
    
    @staticmethod
    def sigmoid(x):
        x = cut_x(x)
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def relu(x):
        x = cut_x(x)
        return x if x > (0) else (0)
    
    @staticmethod
    def gelu(x):
        x = cut_x(x)
        coefficient = math.sqrt(2 / math.pi)
        return 0.5 * x * (1 + math.tanh(coefficient * (x + 0.044715 * math.pow(x, 3))))

    @staticmethod
    def mish(x):
        x = cut_x(x)
        e_x = math.exp(x)
        softplus = math.log(1+e_x, 10)
        mish = x*math.tanh(softplus)
        return mish
    
    @staticmethod
    def no_output_function(x):
        return x
    
    @staticmethod
    def softmax(x):
        x_max = max(x)
        e_x = [math.exp(xi - x_max) for xi in x]
        sum_e_x = sum(e_x)
        softmax = [ex_i / sum_e_x for ex_i in e_x]
        return softmax

    @staticmethod
    def output_sigmoid(x):
        max_x = max(x)
        return [1 / (1 + math.exp(xi - max_x)) for xi in x]
    

    @staticmethod
    def can_finish(row_errors):  # Check if each row has 0 as error
        for d in row_errors:
            if abs(d) > 0.5:
                return False
        return True

    def __str__(self):
        aux = ''
        for ln, layer in enumerate(self.network, start=1):
            aux += '' + '-' * 20 + ' Layer ' + str(ln) + ' ' + '-' * 20 + '\n'
            for neuron in layer:
                aux += str(neuron)
        aux += '-' * 50
        return aux
    
    


# Another functions

def cut_x(x):
    if x > 500:
        x = 500
    elif x < -500:
        x = -500
    return x

def round_array(array, number):
    x = list()
    for i in array:
        x.append(round(i, number))
    return x
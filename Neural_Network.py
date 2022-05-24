import random


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
    def __init__(self):
        self.__ERROR = True
        self.error_rate = 0.01
        self.maximum_restarts = 10
        self.current_restart = 0

        self.network = list()
        self.row_errors = list()

        self.it_n = 0

        self.data_in = list()
        self.data_out = list()

    def set_data(self, data_in, data_out):
        self.data_in = data_in
        self.data_out = data_out
        if len(self.data_in) != len(self.data_out):
            print('Error: the number of rows must be equals for inputs and outputs')
        else:
            self.__ERROR = False

    def make_layers(self, model):
        if self.__ERROR:
            return
        if len(self.data_out[0]) != model[-1]:
            print(f'Error: number of outputs ({len(self.data_out[0])}), expected ({model[-1]})')
            self.__ERROR = True
            return

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

    def predict(self, row, show=None):
        if show is None:
            show = False

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

        if show:
            print(f'Prediction: {predictions}')
        return predictions

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
    def f(num):  # Return 0 if n is less than 0 and 1 otherwise
        return 0 if num < 0 else 1

    @staticmethod
    def can_finish(row_errors):  # Check if each row has 0 as error
        for d in row_errors:
            if d != 0:
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

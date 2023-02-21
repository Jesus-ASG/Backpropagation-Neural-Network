# Import Neural Network
import Neural_Network as Nn

import random

# Explanation:
# In this example we teach to neural network 3 letters wich are A, B and C
# After, let's modify it a bit and see what is his predict


class ExampleRecognizeLetters:
    def __init__(self):
        self.A = [0, 1, 1, 1, 0,
                  1, 0, 0, 0, 1,
                  1, 0, 0, 0, 1,
                  1, 1, 1, 1, 1,
                  1, 0, 0, 0, 1,
                  1, 0, 0, 0, 1,
                  1, 0, 0, 0, 1]

        self.B = [1, 1, 1, 1, 0,
                  1, 0, 0, 0, 1,
                  1, 0, 0, 0, 1,
                  1, 1, 1, 1, 0,
                  1, 0, 0, 0, 1,
                  1, 0, 0, 0, 1,
                  1, 1, 1, 1, 0]

        self.C = [0, 1, 1, 1, 0,
                  1, 0, 0, 0, 1,
                  1, 0, 0, 0, 0,
                  1, 0, 0, 0, 0,
                  1, 0, 0, 0, 0,
                  1, 0, 0, 0, 1,
                  0, 1, 1, 1, 0]

    def main(self):  # Main code here
        # Make data input and output
        data_in = [self.A,  # A
                   self.B,  # B
                   self.C]  # C

        data_out = [
            [0, 0, 1],  # A
            [0, 1, 0],  # B
            [1, 0, 0]  # C
        ]

        # Make an object of Neural Network
        network = Nn.Network(data=(data_in, data_out), model=[5, 10], activation='MISH', output_activation='SOFTMAX')
        # Train with a maximum iterations
        print('Training model...')
        network.train(500)
        print('Model trained!')

        # Optional - show numer of iterations
        print(f'Iterations: {network.it_n}')

        # Start to predict
        print('Ask for A')
        pred = network.predict(self.A)
        print(f'Prediction: {pred}')


        print('Ask for B')
        pred = network.predict(self.B)
        print(f'Prediction: {pred}')


        print('Ask for C')
        pred = network.predict(self.C)
        print(f'Prediction: {pred}')


        # Alter letters
        print('\nChanging letter values')
        self.alter_letter(self.A)
        print('Ask for A')
        pred = network.predict(self.A)
        print(f'Prediction: {pred}')


        self.alter_letter(self.B)
        print('Ask for B')
        pred = network.predict(self.B)
        print(f'Prediction: {pred}')


        self.alter_letter(self.C)
        print('Ask for C')
        pred = network.predict(self.C)
        print(f'Prediction: {pred}')

        # For show all neurons values just use print(network)

        #print(network)

    def alter_letter(self, letter):
        original = self.str_letter(letter)
        m = random.randint(0, len(letter) - 1)
        if letter[m] == 0:
            letter[m] = 1
        else:
            letter[m] = 0
        new_letter = self.str_letter(letter)
        print(f'  Original:       New:')
        original = original.split('\n', 8)
        new_letter = new_letter.split('\n', 8)
        for o, mod in zip(original, new_letter):
            print(f'{o}      {mod}')

    @staticmethod
    def str_letter(letter):  # Return string of letter
        aux = ''
        for i in range(1, len(letter)+1):
            if letter[i-1] == 0:
                aux += '  '
            else:
                aux += '■ '
            if i % 5 == 0:
                aux += '\n'
        return aux


if __name__ == "__main__":
    e = ExampleRecognizeLetters()
    e.main()

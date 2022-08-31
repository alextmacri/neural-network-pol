import numpy as np

class NeuralNetwork:
    def __init__(self, *layers_sizes, name='Neural Network'):
        self.layers = [np.zeros((size, 1)) for size in layers_sizes]
        self.weights = [self.weight_init(layers_sizes[i+1], layers_sizes[i]) for i in range(len(layers_sizes)-1)]
        self.biases = [np.zeros((size, 1)) for size in layers_sizes[1:]]
        self.NAME = name
        
    def __repr__(self):
        representation = 'network:\n'
        for layer in self.layers:
            representation += str(layer.shape[0]) + ' -> '
        representation = representation[:-4] + '\n\n'
        
        representation += '\n\nweights:\n'
        for weight_layer in self.weights:
            representation += str(weight_layer) + '\n\n'
            
        representation += '\n\nbiases:\n'
        for bias_layer in self.biases:
            representation += str(bias_layer) + '\n\n'
            
        return representation[:-2]
    
    def weight_init(self, y, x):
        return np.random.uniform(-0.5, 0.5, (y, x))
    
    def sigmoid_activation_function(self, layer):
        return 1 / (1 + np.exp(layer))
    
    def forward_propagation(self, input_layer):
        self.layers[0] = input_layer
        for i in range(1, len(self.layers)):
            self.layers[i] = self.biases[i-1] + self.weights[i-1] @ self.layers[i-1]
            self.layers[i] = self.sigmoid_activation_function(-self.layers[i])
        return self.layers[-1]
    
    def back_propagation(self, output, label, alpha):
        for i in range(-1, -(len(self.weights)+1), -1):
            if i == -1:
                delta = output - label
            else:
                delta = self.weights[i+1].T @ delta * (self.layers[i] * (1 - self.layers[i]))
            self.weights[i] += -alpha * delta @ self.layers[i-1].T
            self.biases[i] += -alpha * delta
    
    def learn(self, training_data, training_labels, epochs, alpha):
        print(f'Starting to train {self.NAME}')
        total = len(training_data)
        for epoch in range(epochs):
            num_correct = 0
            for data, label in zip(training_data, training_labels):
                data.shape = (data.shape[0], 1)
                label.shape = (label.shape[0], 1)
                
                output = self.forward_propagation(data)
                self.back_propagation(output, label, alpha)
                
                num_correct += int(np.argmax(output) == np.argmax(label))
            
            print(f'epoch {epoch+1}: {num_correct} out of {total} correct')
        print(f'Done training {self.NAME}!')
        
    def test_set(self, test_data, test_labels):
        num_correct = 0
        for data, label in zip(test_data, test_labels):
            data.shape = (data.shape[0], 1)
            label.shape = (label.shape[0], 1)
            
            output = self.forward_propagation(data)
            num_correct += int(np.argmax(output) == np.argmax(label))
            
        print('{}\'s score: {:.2f}%'.format(self.NAME, num_correct/len(test_data)*100))
        
    def test_single(self, data, label):
        print(f'guess: {self.forward_propagation(data).argmax()}\nanswer: {label.argmax()}')

    def save_to_file(self, path_weights, path_biases):
        np.save(path_weights, self.weights)
        np.save(path_biases, self.biases)

    def load_from_file(self, path_weights, path_biases):
        self.weights = np.load(path_weights, allow_pickle=True)
        self.biases = np.load(path_biases, allow_pickle=True)


if __name__ == '__main__':
    with np.load(f'data/mnist.npz') as f:
        training_images = f['x_train'].astype('float32') / 255    # loading and regularizing
        new_y, new_x = training_images.shape[0], training_images.shape[1] * training_images.shape[2]
        training_images = np.reshape(training_images, (new_y, new_x))    # making each 2D image array into a 1D array
        
        training_labels = f['y_train']
        training_labels = np.eye(10)[training_labels]    # shortcut to one hot encoding the labels
        
        test_images = f['x_test'].astype('float32') / 255    # repeating the previous steps for the test data
        new_y, new_x = test_images.shape[0], test_images.shape[1] * test_images.shape[2]
        test_images = np.reshape(test_images, (new_y, new_x))
        
        test_labels = f['y_test']
        test_labels = np.eye(10)[test_labels]

    NN = NeuralNetwork(784, 20, 10)
    # NN.learn(training_images, training_labels, 5, 0.01)
    NN.load_from_file('live_demo_weights.npy', 'live_demo_biases.npy')

    NN.test_set(test_images, test_labels)

    index = np.random.randint(len(test_images))
    image, label = test_images[index], test_labels[index]
    print(image.shape)
    image.shape = (image.shape[0], 1)
    NN.test_single(image, label)

    input('Press Enter to save trained neural network model.')
    NN.save_to_file('live_demo_weights.npy', 'live_demo_biases.npy')
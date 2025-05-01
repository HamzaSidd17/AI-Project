import numpy as np

class neuralNet:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]        
        self.input_vector = []
        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            w = np.random.uniform(-1, 1, (self.layer_sizes[i + 1], self.layer_sizes[i]))
            b = np.random.uniform(-1, 1, (self.layer_sizes[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def set_input_vector(self, angle, trackPos, speedX, rpm, gear, trackSensors):
        self.input_vector = [
            angle,
            trackPos,
            speedX / 300.0,
            rpm / 9000.0,
            gear / 6.0
        ]

        # Normalize track sensors to [0, 1]
        normalized_sensors = [min(s / 200.0, 1.0) for s in trackSensors]
        self.input_vector.extend(normalized_sensors)

    def activation_sigmoid(self, val):
        return 1 / (1 + np.exp(-val))

    def feed_forward(self):
        accel = None
        steer = None
        brake = None 
        a = np.array(self.input_vector).reshape(-1, 1)
        # print("input vector: ", a)

        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            # a = np.tanh(z)  # Hidden activation
            a = self.activation_sigmoid(z)

        z = np.dot(self.weights[-1], a) + self.biases[-1]
        output = z.flatten()  
        
        # Post-processing
        steer = np.tanh(output[0])  # [0 = 1] continuous 
        accel = self.activation_sigmoid(output[1])          # [0, 1] discrete
        brake = self.activation_sigmoid(output[2])          # [0, 1] discrete

        # print("output vector: ", steer, accel, brake)

        return accel, steer, brake 
    
    def get_genome(self):
        """
        Flattens weights and biases into a single genome array.
        """
        genome = []
        for w, b in zip(self.weights, self.biases):
            genome.extend(w.flatten())
            genome.extend(b.flatten())
        return np.array(genome)

    def set_genome(self, genome):
        """
        Loads weights and biases from a flat genome array.
        """
        idx = 0
        for i in range(len(self.weights)):
            w_shape = self.weights[i].shape
            b_shape = self.biases[i].shape
            w_size = np.prod(w_shape)
            b_size = np.prod(b_shape)

            self.weights[i] = genome[idx:idx + w_size].reshape(w_shape)
            idx += w_size

            self.biases[i] = genome[idx:idx + b_size].reshape(b_shape)
            idx += b_size

    def genome_fitness(self, state):
        fitness = 100
        for sensor in state.track:
            if sensor <= 0:
                fitness -= 10

        if state.racePos > 5:
            fitness -= (state.racePos - 1) * 10
        else: 
            fitness += (state.racePos - 1) * 10
        
        if state.speedX > 300:
            fitness += 10

        fitness = self.activation_sigmoid(fitness)

        

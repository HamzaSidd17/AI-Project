import numpy as np
from math import *
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
        # Distance to the edge of the track is between 0 and 200
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
    # Load gennome from a file
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

    
    def fitness(self, state, lapCompleted=False):
        """
        Calculates the fitness of the genome based on the state of the car.
        
        Parameters:
        - state: The current state of the car with various attributes
        - lapCompleted: Boolean indicating if a lap was completed
        
        Returns:
        - fitness: A numerical value representing the performance
        """
        # Base fitness starts at zero
        fitness = 0.0
        
        # === PROGRESS REWARD ===
        track_length = state.trackLength if hasattr(state, 'trackLength') else 7000  # Default if not available
        progress_reward = (state.distanceFromStart / track_length) * 1000
        fitness += progress_reward
        
        # === SPEED REWARD ===
        # Reward for maintaining good speed (up to 300 km/h with diminishing returns after that)
        optimal_speed = 250  # km/h
        speed_factor = min(state.speedX / optimal_speed, 1.2)  # Allow up to 30% over optimal
        speed_reward = 200 * speed_factor
        fitness += speed_reward
        
        # === TRACK POSITION PENALTY ===
        track_penalty = 0
        sensors_off_track = sum(1 for sensor in state.track if sensor <= 0)
        if sensors_off_track > 0:
            # Severe penalty for going off track
            track_penalty = 500 * sensors_off_track
        else:
            # Smaller penalty for being close to edge (trackPos ranges from -1 to 1)
            edge_proximity = abs(state.trackPos)
            if edge_proximity > 0.7:  # If close to edge
                track_penalty = 100 * (edge_proximity - 0.7) / 0.3
        fitness -= track_penalty
        
        # === RACE POSITION REWARD ===
        position_factor = max(1, 10 - state.racePos)  # Higher reward for positions 1-10
        position_reward = position_factor * 50
        fitness += position_reward
        
        # === LAP TIME REWARD/PENALTY ===
        # Handle lap times
        lap_time_factor = 0
        
        if lapCompleted:
            # Big reward for completing a lap
            fitness += 2000
            
            # Additional reward for lap time improvement
            if state.lastLapTime is not None and state.lastLapTime > 0:
                previous_lap = state.lastLapTime
                current_lap = state.currentLapTime
                
                # Calculate improvement percentage
                if current_lap < previous_lap:
                    improvement = (previous_lap - current_lap) / previous_lap
                    lap_time_factor = 1000 * improvement  # Reward proportional to improvement
                else:
                    # Small penalty for regression
                    regression = (current_lap - previous_lap) / previous_lap
                    lap_time_factor = -200 * regression
        else:
            # Small penalty for not completing a lap
            fitness -= 100
            
            # Penalize excessive time without lap completion
            if state.currentLapTime > 120:  # If more than 2 minutes without completion
                time_penalty = min(500, (state.currentLapTime - 120) * 5)
                fitness -= time_penalty
        
        fitness += lap_time_factor
        
        # === STABILITY REWARD ===
        # Optional
        if hasattr(state, 'wheelSpinVel'):
            # Check for wheel spin consistency (traction)
            wheel_variance = np.var(state.wheelSpinVel)
            traction_reward = 100 * (1 - min(1, wheel_variance / 50))
            fitness += traction_reward
        
        return fitness
import numpy as np
import math
import numpy as np
import carState
class neuralNet:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]       
        self.input_vector = []
        self.weights = []
        self.biases = []
        self.fitness_value = None
        self.steer_lock = 0.785398
        self.previous_damage = None
        # Initialize weights with Xavier Uniform
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            w = np.random.uniform(-limit, limit, (fan_out, fan_in))
            self.weights.append(w)

            # Initialize all biases to zero (no special bias for accel/brake)
            b = np.zeros((fan_out, 1))
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
        a = np.array(self.input_vector).reshape(-1, 1)

        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = self.activation_sigmoid(z)

        z = np.dot(self.weights[-1], a) + self.biases[-1]
        output = z.flatten()  
        
        # Post-processing
        steer = np.tanh(output[0])        # [-1, 1] continuous 
        accel = self.activation_sigmoid(output[1])  # [0, 1] continuous
        
        brake =  self.activation_sigmoid(output[2]) # Scale down brake values
        brake = 0       
        return accel, steer, brake 
    
    def get_genome(self):
        genome = []
        for w, b in zip(self.weights, self.biases):
            genome.extend(w.flatten())
            genome.extend(b.flatten())
        return np.array(genome)

    def set_genome(self, genome):
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            w_size = in_size * out_size
            b_size = out_size

            self.weights[i] = genome[idx:idx + w_size].reshape((out_size, in_size))
            idx += w_size
            self.biases[i] = genome[idx:idx + b_size].reshape((out_size, 1))
            idx += b_size

    def fitness(self, state, output, lapCompleted=False):
        fitnessV = 0.0

        # ========== Constants ==========
        BASE_REWARDS = {
            'distance': 5.0,        # Per meter raced
            'track_center': 15.0,   # Max reward for perfect center
            'speed_optimal': 10.0,   # Reward for maintaining ideal speed
            'steering_alignment': 8.0, # For matching desired steering
            'cornering_efficiency': 12.0, # For smooth turns
            'stability': 7.0,        # For low sliding/rotation
            'damage_penalty': 25.0,  # Per damage point
            'offtrack_penalty': 100.0, # Immediate penalty
            'gear_efficiency': 3.0,  # For proper RPM management
            'lap_completion': 5000.0 # Big bonus for completing laps
        }

        # ========== Core Components ==========
        
        #1. Distance Progress (Primary Reward)
        fitnessV += state.distRaced * BASE_REWARDS['distance']
        
        # 2. Track Position (Bell Curve Reward)
        track_pos_score = math.exp(-4 * (state.trackPos ** 2))
        fitnessV += BASE_REWARDS['track_center'] * track_pos_score
        
        # 3. Dynamic Speed Control
        max_safe_speed = self.calculate_max_safe_speed(state.track)
        speed_ratio = min(state.speedX / max_safe_speed, 1.5)
        fitnessV += BASE_REWARDS['speed_optimal'] * speed_ratio
        
        # # 4. Steering Quality
        desired_steer = self.get_desired_steering(state)
        steer_error = abs(desired_steer - output[1])
        steer_score = 1.0 / (1.0 + 2.0 * steer_error)
        fitnessV += BASE_REWARDS['steering_alignment'] * steer_score
        
        # 5. Cornering Efficiency
        turn_quality = (1.0 - abs(state.angle)) * (1.0 - abs(state.speedY)/15.0)
        fitnessV += BASE_REWARDS['cornering_efficiency'] * turn_quality
        
        if self.previous_damage == None:
            self.previous_damage = state.damage
        # # ========== Penalties ==========
        
        # # Off-track penalty (exponential)
        if abs(state.trackPos) > 1.0:
            fitnessV -= BASE_REWARDS['offtrack_penalty'] * (abs(state.trackPos) - 1.0)
        
        # Damage penalty
        min_wall_distance = min(state.track)  # Use track sensors
        CRASH_PENALTY = -3000  # Immediate heavy penalty for wall contact
        NEAR_MISS_PENALTY = -5000 * (1 - min_wall_distance)  # Progressive penalty
        SPEED_CRASH_MULTIPLIER = 2.0  # Penalize high-speed crashes more

        if state.damage > self.previous_damage:  # Detect new collision
            fitnessV += CRASH_PENALTY * (1 + state.speedX * SPEED_CRASH_MULTIPLIER)
            self.previous_damage = state.damage  # Update previous damage
        # Progressive penalty for dangerous proximity
        if min_wall_distance and min_wall_distance < 5:  # 5 meters from wall
            fitnessV += NEAR_MISS_PENALTY * (5 - min_wall_distance)

        
        # Oversteer/Understeer detection
        # if abs(state.angle) > 0.5 and abs(state.speedY) > 10:
        #     fitnessV -= 30.0  # Severe sliding penalty
        
            
        # ========== Special Bonuses ==========
        
        # Complete lap bonus
        if lapCompleted:
            fitnessV += BASE_REWARDS['lap_completion']
            
        # Speed maintenance bonus
        # if state.speedX > 0.9 * max_safe_speed:
        #     fitnessV += 25.0
            
        # Perfect corner bonus
        if steer_error < 0.1 and abs(state.angle) < 0.2:
            fitnessV += 20.0

        self.fitness_value = max(fitnessV, 0)  # Prevent negative scores
        return self.fitness_value

    def calculate_max_safe_speed(self, track_sensors):
        """Calculate maximum safe speed based on upcoming track curvature"""
        # Look at sensors 5-15 (front 100 degree arc)
        front_sensors = track_sensors[5:15]
        min_distance = min(front_sensors)
        
        # Dynamic speed calculation
        if min_distance < 20:
            return 60  # Slow for tight corners
        elif min_distance < 50:
            return 120
        return 250  # Straightaway speed

    def get_desired_steering(self, state):
        """Calculate ideal steering using sensor fusion"""
        # Weighted average of multiple steering strategies
        track_based = (state.angle - state.trackPos * 0.5) / self.steer_lock
        lookahead = self.calculate_lookahead_steering(state.track)
        return 0.7 * track_based + 0.3 * lookahead

    def calculate_lookahead_steering(self, track):
        """Anticipate upcoming turns using front sensors"""
        # Emphasize sensors 8-10 (central front)
        curve_indicator = track[8] + track[9] + track[10]
        left_indicator = sum(track[0:5])
        right_indicator = sum(track[14:19])
        
        if left_indicator > right_indicator + 10:
            return -0.4  # Prepare right turn
        elif right_indicator > left_indicator + 10:
            return 0.4   # Prepare left turn
        return 0.0




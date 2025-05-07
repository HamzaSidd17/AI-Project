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
        self.prev_rpm = 0
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
        """
        Enhanced fitness function for racing car with better handling of track limits, 
        acceleration, braking, and other racing characteristics.
        
        Args:
            state: CarState object containing all car sensors and state information
            output: Tuple of (accel, steer, brake) values produced by the neural network
            lapCompleted: Boolean indicating if a lap was completed in this step
            
        Returns:
            float: The calculated fitness value
        """
        fitnessV = 0.0
        
        # ========== Constants ==========
        BASE_REWARDS = {
            'distance': 8.0,             # Per meter raced (increased for faster progress)
            'track_center': 15.0,        # Max reward for staying near track center
            'speed_optimal': 12.0,       # Reward for maintaining ideal speed
            'steering_alignment': 10.0,  # For matching desired steering
            'cornering_efficiency': 15.0,# For smooth turns
            'stability': 10.0,           # For low sliding/rotation
            'braking_efficiency': 8.0,   # For proper braking before corners
            'acceleration_efficiency': 8.0, # For proper acceleration out of corners
            'gear_efficiency': 5.0,      # For proper RPM management
            'track_progression': 20.0,   # For consistent forward progress
            'damage_penalty': 50.0,      # Per damage point (increased penalty)
            'offtrack_penalty': 200.0,   # Immediate penalty (doubled)
            'lap_completion': 8000.0     # Big bonus for completing laps (increased)
        }
        
        # Initialize state tracking if first call
        if not hasattr(self, 'previous_state'):
            self.previous_state = {
                'damage': state.damage,
                'distRaced': state.distRaced,
                'speedX': state.speedX,
                'position': state.trackPos,
                'time': state.curLapTime or 0.0,
                'rpm': state.rpm,
                'offtrack_count': 0,
                'recovery_bonus_applied': False,
                'corner_count': 0,
                'last_corner_entry': 0,
                'accel': output[0],
                'brake': output[2] if len(output) > 2 else 0
            }
        
        # ========== Primary Rewards ==========
        
        # 1. Distance Progress (Primary Reward)
        distance_progress = state.distRaced - self.previous_state['distRaced']
        if distance_progress > 0:
            fitnessV += distance_progress * BASE_REWARDS['distance']
        
        # Track progression bonus
        if state.distRaced > self.previous_state['distRaced']:
            # Extra reward for consistent forward progress
            progression_rate = distance_progress / (state.curLapTime - self.previous_state['time'] + 0.001)
            fitnessV += progression_rate * BASE_REWARDS['track_progression']
        
        # 2. Track Position (Advanced Bell Curve Reward)
        # More forgiving near center, steeper penalty near edges
        track_pos_value = abs(state.trackPos)
        if track_pos_value <= 0.5:
            # Inner half of track - high reward zone
            track_pos_score = 1.0 - 0.5 * (track_pos_value ** 2)
        else:
            # Outer half of track - rapidly decreasing reward
            track_pos_score = 0.75 * math.exp(-3 * (track_pos_value - 0.5))
        
        fitnessV += BASE_REWARDS['track_center'] * track_pos_score
        
        # 3. Dynamic Speed Control with curve analysis
        max_safe_speed = self.calculate_max_safe_speed(state.track)
        speed_ratio = min(state.speedX / max_safe_speed, 1.2)  # Cap at 120% of max safe speed
        
        # Progressive speed reward based on context
        if speed_ratio < 0.7:
            # Too slow - gradually increasing reward
            speed_score = speed_ratio / 0.7
        elif speed_ratio <= 1.0:
            # Optimal speed zone - max reward
            speed_score = 1.0
        else:
            # Too fast - exponentially decreasing reward
            speed_score = math.exp(-2 * (speed_ratio - 1.0))
        
        fitnessV += BASE_REWARDS['speed_optimal'] * speed_score
        
        # 4. Steering Quality
        desired_steer = self.get_desired_steering(state)
        steer_error = abs(desired_steer - output[1])
        
        # Progressive steering reward - more sensitive to small errors
        if steer_error < 0.1:
            steer_score = 1.0 - steer_error  # Linear decrease for small errors
        else:
            steer_score = 0.9 * math.exp(-2 * steer_error)  # Exponential decrease for larger errors
        
        fitnessV += BASE_REWARDS['steering_alignment'] * steer_score
        
        # 5. Cornering Efficiency with advanced metrics
        # Combine multiple factors: angle, lateral speed, steering stability
        lateral_stability = 1.0 - min(abs(state.speedY) / 20.0, 1.0)
        angle_alignment = 1.0 - min(abs(state.angle) / math.pi, 1.0)
        
        # Detect if we're in a corner
        in_corner = self.is_in_corner(state)
        if in_corner:
            # We're in a corner - evaluate cornering technique
            corner_score = 0.6 * lateral_stability + 0.4 * angle_alignment
            
            # Extra reward for smooth corner entry
            if self.previous_state['corner_count'] == 0:
                # Just entered a corner
                self.previous_state['corner_count'] = 1
                self.previous_state['last_corner_entry'] = state.distRaced
                
                # Check if braking was applied before corner
                if self.previous_state['brake'] > 0.2:
                    fitnessV += 15.0  # Reward for proper corner preparation
                    
            # Check corner exit
            elif state.distRaced - self.previous_state['last_corner_entry'] > 20:
                # Exiting corner - reward acceleration
                if output[0] > 0.5 and lateral_stability > 0.7:
                    fitnessV += 20.0  # Reward for good corner exit
                self.previous_state['corner_count'] = 0
        else:
            # We're on a straight - different metrics
            corner_score = 0.8 * angle_alignment + 0.2 * lateral_stability
            self.previous_state['corner_count'] = max(0, self.previous_state['corner_count'] - 0.1)
        
        fitnessV += BASE_REWARDS['cornering_efficiency'] * corner_score
        
        # 6. Acceleration and Braking Efficiency
        # Calculate track curvature ahead and behind
        ahead_curve = self.calculate_upcoming_curvature(state.track)
        behind_curve = self.calculate_behind_curvature(state.track)
        
        # Evaluate acceleration based on context
        accel = output[0]
        brake = output[2] if len(output) > 2 else 0
        
        # Intelligent acceleration scoring
        if ahead_curve > 0.7:  # Sharp curve ahead
            # Should be slowing down
            accel_score = 1.0 - accel
        elif ahead_curve > 0.3:  # Moderate curve ahead
            # Partial acceleration is appropriate
            accel_score = 1.0 - abs(0.5 - accel)
        else:  # Straight or gentle curve
            # Full acceleration is appropriate
            accel_score = accel
        
        # Intelligent braking scoring
        if ahead_curve > 0.7:  # Sharp curve ahead
            # Should be braking
            brake_score = brake
        elif ahead_curve > 0.3:  # Moderate curve ahead
            # Light braking may be appropriate
            brake_score = 1.0 - abs(0.3 - brake)
        else:  # Straight or gentle curve
            # No braking needed
            brake_score = 1.0 - brake
        
        # Apply rewards
        fitnessV += BASE_REWARDS['acceleration_efficiency'] * accel_score
        fitnessV += BASE_REWARDS['braking_efficiency'] * brake_score
        
        # 7. Gear and RPM Efficiency
        rpm_efficiency = 1.0 - abs((state.rpm - 6000) / 10000)  # Optimal RPM around 6000
        fitnessV += BASE_REWARDS['gear_efficiency'] * rpm_efficiency
        
        # ========== Penalties ==========
        
        # 1. Off-track penalty (progressive and persistent)
        if abs(state.trackPos) > 1.0:
            # Car is off-track - severe penalty
            off_track_factor = abs(state.trackPos) - 1.0
            fitnessV -= BASE_REWARDS['offtrack_penalty'] * (1.0 + off_track_factor * off_track_factor)
            
            # Increase off-track counter for persistent penalty
            self.previous_state['offtrack_count'] += 1
            
            # Extra penalty for staying off-track
            fitnessV -= 10.0 * self.previous_state['offtrack_count']
            
            # Reset recovery bonus flag
            self.previous_state['recovery_bonus_applied'] = False
        else:
            # Car is on track - reset counter with hysteresis
            if self.previous_state['offtrack_count'] > 0:
                self.previous_state['offtrack_count'] = max(0, self.previous_state['offtrack_count'] - 0.5)
                
                # Apply recovery bonus once when returning to track
                if not self.previous_state['recovery_bonus_applied'] and abs(state.trackPos) < 0.8:
                    fitnessV += 100.0  # Recovery bonus
                    self.previous_state['recovery_bonus_applied'] = True
        
        # 2. Damage penalty (enhanced)
        if state.damage > self.previous_state['damage']:
            # New damage occurred
            damage_delta = state.damage - self.previous_state['damage']
            base_damage_penalty = BASE_REWARDS['damage_penalty'] * damage_delta
            
            # Apply speed multiplier for high-speed impacts
            speed_factor = 1.0 + max(0, state.speedX / 100.0)
            total_damage_penalty = base_damage_penalty * speed_factor
            
            fitnessV -= total_damage_penalty
            self.previous_state['damage'] = state.damage
        
        # 3. Wall proximity penalty (proactive avoidance)
        min_wall_distance = min(state.track) if state.track else 200
        
        # Progressive penalty increasing as wall gets closer
        if min_wall_distance < 15:
            wall_danger = (15.0 - min_wall_distance) / 15.0
            wall_penalty = 200.0 * (wall_danger ** 2)
            
            # Additional penalty for approaching walls at high speed
            if state.speedX > 80:
                speed_danger = (state.speedX - 80) / 100.0
                wall_penalty *= (1.0 + speed_danger)
                
            fitnessV -= wall_penalty
        
        # 4. Sliding/Instability penalty
        lateral_speed_penalty = 0
        if abs(state.speedY) > 5:
            # Car is sliding sideways
            slide_factor = (abs(state.speedY) - 5) / 15.0
            lateral_speed_penalty = 50.0 * slide_factor * slide_factor
            fitnessV -= lateral_speed_penalty
        
        # 5. Wrong direction penalty
        if state.angle > math.pi/2 or state.angle < -math.pi/2:
            # Car is pointing in wrong direction (more than 90 degrees off)
            wrong_direction_penalty = 500.0
            fitnessV -= wrong_direction_penalty
        
        # ========== Special Bonuses ==========
        
        # 1. Complete lap bonus with time-based multiplier
        if lapCompleted:
            # Base completion bonus
            lap_bonus = BASE_REWARDS['lap_completion']
            
            # Time-based multiplier (better time = higher bonus)
            if state.lastLapTime and state.lastLapTime > 0:
                # Assuming a good lap is around 90 seconds (adjust based on track)
                time_factor = min(120 / state.lastLapTime, 2.0)
                lap_bonus *= time_factor
                
            fitnessV += lap_bonus
        
        # 2. Perfect corner bonus (refined criteria)
        corner_precision = (steer_score > 0.9) and (lateral_stability > 0.85) and (angle_alignment > 0.9)
        if corner_precision and in_corner:
            fitnessV += 30.0  # Significant bonus for perfect cornering
        
        # 3. Racing line bonus
        # Reward maintaining ideal racing line (shifting from outside to inside to outside in corners)
        if in_corner and self.previous_state['corner_count'] > 0:
            # Check if racing line is being followed based on track position changes
            racing_line_bonus = 0
            
            # Simple heuristic for racing line - more sophisticated analysis could be implemented
            if (state.trackPos - self.previous_state['position']) * state.angle < 0:
                # Track position is changing in right direction compared to car angle
                racing_line_bonus = 15.0
            
            fitnessV += racing_line_bonus
        
        # 4. Consistent speed maintenance bonus
        if abs(state.speedX - self.previous_state['speedX']) < 5.0 and state.speedX > 0.8 * max_safe_speed:
            # Maintaining steady high speed
            fitnessV += 10.0
            
        # Update previous state for next iteration
        self.previous_state['distRaced'] = state.distRaced
        self.previous_state['speedX'] = state.speedX
        self.previous_state['position'] = state.trackPos
        self.previous_state['time'] = state.curLapTime or self.previous_state['time']
        self.previous_state['rpm'] = state.rpm
        self.previous_state['accel'] = output[0]
        self.previous_state['brake'] = output[2] if len(output) > 2 else 0

        # Ensure fitness doesn't go below a minimum threshold to prevent genetic stagnation
        self.fitness_value = max(fitnessV, 10.0)  
        return self.fitness_value
    
    def get_ideal_gear(self, speedX):
        if speedX > 170:
            return 6
        elif speedX > 140:
            return 5
        elif speedX > 110:
            return 4
        elif speedX > 80:
            return 3
        elif speedX > 50:
            return 2
        else:
            return 1

    # call this function to get gear
    def gear(self, rpm, speedX, current_gear):
        rpm_rising = (rpm - self.prev_rpm) > 0
        self.prev_rpm = rpm  

        target_gear = self.get_ideal_gear(speedX)

        if rpm_rising and rpm > 7000:
            target_gear = max(current_gear + 1, target_gear)
        elif not rpm_rising and rpm < 3000:
            target_gear = min(current_gear - 1, target_gear)

        target_gear = max(1, min(6, target_gear))

        return target_gear

    def calculate_max_safe_speed(self, track_sensors):
        """
        Calculate maximum safe speed based on upcoming track curvature with more detailed analysis
        """
        # Look at sensors 3-15 for wider field of view (more anticipation)
        front_sensors = track_sensors[3:16]
        
        # Calculate average distance and minimum distance
        avg_distance = sum(front_sensors) / len(front_sensors)
        min_distance = min(front_sensors)
        
        # Calculate variance in distances (indicates curvature)
        variance = sum((d - avg_distance)**2 for d in front_sensors) / len(front_sensors)
        std_dev = math.sqrt(variance)
        
        # Use both minimum distance and standard deviation to estimate curvature
        curvature_factor = (min_distance / 200.0) * (1.0 - min(std_dev / 50.0, 0.5))
        
        # Dynamic speed calculation - smoother transition
        if curvature_factor < 0.3:
            # Very sharp corner
            return 50 + 100 * curvature_factor  # 50-80 km/h
        elif curvature_factor < 0.6:
            # Moderate corner
            return 80 + 150 * (curvature_factor - 0.3) / 0.3  # 80-130 km/h
        else:
            # Gentle curve or straight
            return 130 + 150 * (curvature_factor - 0.6) / 0.4  # 130-280 km/h

    def get_desired_steering(self, state):
        """
        Calculate ideal steering using enhanced sensor fusion and prediction
        """
        # Basic steering component based on track position and angle
        position_correction = -state.trackPos * 0.7
        angle_correction = state.angle / self.steer_lock
        
        # Lookahead component for anticipation
        lookahead = self.calculate_lookahead_steering(state.track)
        
        # Dynamic weighting based on speed
        speed_factor = min(state.speedX / 150.0, 1.0)
        
        # At higher speeds, increase weight of lookahead component
        position_weight = 0.6 - 0.3 * speed_factor
        angle_weight = 0.6 - 0.1 * speed_factor
        lookahead_weight = 0.3 + 0.4 * speed_factor
        
        # Combined steering strategy with adaptive weights
        return (position_correction * position_weight + 
                angle_correction * angle_weight + 
                lookahead * lookahead_weight)

    def calculate_lookahead_steering(self, track):
        """
        Enhanced lookahead steering with better curve detection
        """
        # Create segments for different areas
        center_sensors = track[8:11]  # Central front
        left_sensors = track[0:6]     # Left side
        right_sensors = track[13:19]  # Right side
        
        # Calculate averages and find minimum distances
        center_min = min(center_sensors)
        left_min = min(left_sensors)
        right_min = min(right_sensors)
        
        left_avg = sum(left_sensors) / len(left_sensors)
        right_avg = sum(right_sensors) / len(right_sensors)
        
        # Check for significant differences indicating curves
        if left_avg > right_avg * 1.5:
            # Right curve ahead - steer right with adaptive intensity
            intensity = min(0.8, (left_avg - right_avg) / 100.0)
            return intensity  # Positive is right
        elif right_avg > left_avg * 1.5:
            # Left curve ahead - steer left with adaptive intensity
            intensity = min(0.8, (right_avg - left_avg) / 100.0)
            return -intensity  # Negative is left
        elif center_min < 50 and (left_min > center_min * 1.3 or right_min > center_min * 1.3):
            # Obstacle or sharp turn directly ahead
            if left_min > right_min:
                return -0.5  # Turn left
            else:
                return 0.5   # Turn right
        else:
            # No significant curve detected
            return 0.0

    def is_in_corner(self, state):
        """
        Determine if the car is currently in a corner
        """
        # Combine multiple indicators
        track_variance = self.calculate_track_variance(state.track)
        speed_recommended = self.calculate_max_safe_speed(state.track)
        
        # Corner indicators:
        # 1. High track sensor variance (walls at different distances)
        # 2. Recommended speed significantly lower than straightaway speed
        # 3. Car has significant angle relative to track
        in_corner = (track_variance > 1500 or 
                    speed_recommended < 180 or 
                    abs(state.angle) > 0.2)
                    
        return in_corner

    def calculate_track_variance(self, track):
        """
        Calculate variance in track sensors as a measure of curvature
        """
        if not track:
            return 0
            
        front_sensors = track[5:14]  # Front sensors
        avg = sum(front_sensors) / len(front_sensors)
        variance = sum((d - avg)**2 for d in front_sensors)
        return variance

    def calculate_upcoming_curvature(self, track):
        """
        Calculate the curvature of the upcoming track section
        Returns value between 0 (straight) and 1 (sharp curve)
        """
        if not track:
            return 0
            
        # Look at forward sensors
        forward_sensors = track[5:14]
        
        # Calculate variance and standard deviation
        avg = sum(forward_sensors) / len(forward_sensors)
        variance = sum((d - avg)**2 for d in forward_sensors) / len(forward_sensors)
        std_dev = math.sqrt(variance)
        
        # Normalize to 0-1 range (0=straight, 1=sharp curve)
        curvature = min(std_dev / 60.0, 1.0)
        
        # Also consider minimum distance
        min_distance = min(forward_sensors)
        distance_factor = 1.0 - min(min_distance / 100.0, 1.0)
        
        # Combine both metrics
        return 0.7 * curvature + 0.3 * distance_factor

    def calculate_behind_curvature(self, track):
        """
        Estimate curvature of the track section behind the car
        Used to evaluate if we're exiting a corner
        """
        if not track:
            return 0
            
        # Look at outer sensors which partially see behind
        side_sensors = track[0:3] + track[16:19]
        
        # Calculate variance
        if len(side_sensors) > 0:
            avg = sum(side_sensors) / len(side_sensors)
            variance = sum((d - avg)**2 for d in side_sensors) / len(side_sensors)
            std_dev = math.sqrt(variance)
            
            # Normalize to 0-1 range
            curvature = min(std_dev / 60.0, 1.0)
            return curvature
        return 0



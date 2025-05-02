import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# This simulates the car state that would come from your racing environment
class CarState:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Import your neural network class

import neuralNet
# from neural_network import neuralNet  # Uncomment this when using the actual file

# Create a test function
def test_neural_network():
    # Create a neural network with appropriate architecture
    # Input: angle, trackPos, speedX, rpm, gear, and track sensors (assuming 19 sensors)
    input_size = 24  # 5 basic inputs + 19 track sensors
    hidden_layers = [16, 8]
    output_size = 3  # accel, steer, brake
    
    nn = neuralNet.neuralNet(input_size, hidden_layers, output_size)
    
    # Test cases for evaluation
    test_cases = [
        # Test case 1: Ideal driving conditions
        {
            'name': 'Ideal Driving',
            'state': CarState(
                angle=0.0,
                trackPos=0.0,  # Center of track
                speedX=250.0,  # Good speed
                rpm=7000.0,
                gear=5,
                track=[200.0] * 19,  # All sensors detect track
                distanceFromStart=5000.0,
                trackLength=7000.0,
                currentLapTime=60.0,
                lastLapTime=65.0,  # Improving lap time
                racePos=1,
                wheelSpinVel=[65.0, 65.0, 65.0, 65.0],
                damage=0
            ),
            'lapCompleted': True
        },
        
        # Test case 2: Off-track situation
        {
            'name': 'Off Track',
            'state': CarState(
                angle=0.5,  # Angled away from track
                trackPos=0.95,  # Near edge
                speedX=200.0,
                rpm=6000.0,
                gear=4,
                track=[50.0, 30.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0],
                distanceFromStart=2000.0,
                trackLength=7000.0,
                currentLapTime=120.0,
                lastLapTime=None,
                racePos=8,
                wheelSpinVel=[40.0, 60.0, 40.0, 60.0],  # Inconsistent wheel spin
                damage=1000
            ),
            'lapCompleted': False
        },
        
        # Test case 3: Slow but safe driving
        {
            'name': 'Slow but Safe',
            'state': CarState(
                angle=0.1,
                trackPos=0.1,  # Good track position
                speedX=150.0,  # Slower than optimal
                rpm=4000.0,
                gear=3,
                track=[180.0] * 19,  # Good track awareness
                distanceFromStart=3000.0,
                trackLength=7000.0,
                currentLapTime=90.0,
                lastLapTime=88.0,  # Slightly worse than last lap
                racePos=5,
                wheelSpinVel=[45.0, 45.0, 45.0, 45.0],  # Consistent
                damage=0
            ),
            'lapCompleted': False
        },
        
        # Test case 4: Fast but risky driving
        {
            'name': 'Fast but Risky',
            'state': CarState(
                angle=0.2,
                trackPos=0.7,  # Close to edge
                speedX=320.0,  # Very fast
                rpm=8500.0,
                gear=6,
                track=[100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
                distanceFromStart=6500.0,
                trackLength=7000.0,
                currentLapTime=50.0,
                lastLapTime=55.0,  # Improved lap time
                racePos=2,
                wheelSpinVel=[80.0, 80.0, 80.0, 80.0],
                damage=200
            ),
            'lapCompleted': True
        },
        
        # Test case 5: Beginning of race
        {
            'name': 'Race Start',
            'state': CarState(
                angle=0.0,
                trackPos=0.0,
                speedX=100.0,  # Accelerating
                rpm=3000.0,
                gear=2,
                track=[200.0] * 19,
                distanceFromStart=500.0,
                trackLength=7000.0,
                currentLapTime=10.0,
                lastLapTime=None,  # First lap
                racePos=10,
                wheelSpinVel=[30.0, 30.0, 30.0, 30.0],
                damage=0
            ),
            'lapCompleted': False
        }
    ]
    
    # Run tests and collect results
    results = []
    
    for test_case in test_cases:
        # Set the input vector
        nn.set_input_vector(
            test_case['state'].angle,
            test_case['state'].trackPos,
            test_case['state'].speedX,
            test_case['state'].rpm,
            test_case['state'].gear,
            test_case['state'].track
        )
        
        # Get actions from neural network
        accel, steer, brake = nn.feed_forward()
        
        # Calculate fitness
        fitness = nn.fitness(test_case['state'], test_case['lapCompleted'])
        
        # Store results
        results.append({
            'name': test_case['name'],
            'fitness': fitness,
            'actions': (accel, steer, brake)
        })
    
    # Print results
    print("\n===== NEURAL NETWORK TEST RESULTS =====")
    for result in results:
        print(f"\nTest Case: {result['name']}")
        print(f"Fitness Score: {result['fitness']:.2f}")
        print(f"Actions (accel, steer, brake): ({result['actions'][0]:.2f}, {result['actions'][1]:.2f}, {result['actions'][2]:.2f})")
    
    # Optional: Visualize results
    plot_results(results)
    
    return results

def plot_results(results):
    # Plot fitness scores
    plt.figure(figsize=(12, 6))
    
    # Fitness plot
    names = [r['name'] for r in results]
    fitness_values = [r['fitness'] for r in results]
    
    plt.subplot(1, 2, 1)
    plt.bar(names, fitness_values)
    plt.title('Fitness Scores by Test Case')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Fitness Score')
    plt.tight_layout()
    
    # Actions plot
    plt.subplot(1, 2, 2)
    accel_values = [r['actions'][0] for r in results]
    steer_values = [r['actions'][1] for r in results]
    brake_values = [r['actions'][2] for r in results]
    
    x = np.arange(len(names))
    width = 0.25
    
    plt.bar(x - width, accel_values, width, label='Acceleration')
    plt.bar(x, steer_values, width, label='Steering')
    plt.bar(x + width, brake_values, width, label='Braking')
    
    plt.title('Network Actions by Test Case')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Action Value')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('neural_network_test_results.png')
    plt.show()

# Main function to run tests
def main():
    test_neural_network()

if __name__ == "__main__":
    main()
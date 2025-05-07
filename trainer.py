import numpy as np
import random
import csv
import os
from neuralNet import neuralNet

class Trainer:
    def __init__(self, population_size=50, mutation_rate=0.3, elite_size=5):
        # Ensure minimum population size for tournament selection
        self.population_size = max(1, population_size)
        self.mutation_rate = mutation_rate
        # Ensure elite_size is reasonable
        self.elite_size = min(elite_size, int(0.1 * self.population_size))
        self.generation = 0
        self.best_fitness_history = []
        self.best_genome_history = []
        self.stagnation_counter = 0  # Track generations without improvement
        self.top_genomes = []  # Store top genomes for initialization
        # Initialize population
        self.population = []
        self.fitness_scores = [0.0] * self.population_size
        for i in range(self.population_size):
            nn = neuralNet(24, [16, 8], 3)
            nn._index = i
            nn._trainer = self
            self.population.append(nn)
    
    def initialize_population(self, input_size, hidden_sizes, output_size, best_genome=None):
        """Create initial population, cloning best_genome if provided."""
        population = []
        for i in range(self.population_size):
            if best_genome is not None and i == 0:
                # Elitism: reuse best genome
                nn = neuralNet(input_size, hidden_sizes, output_size)
                nn.set_genome(np.copy(best_genome))
                population.append(nn)
            elif best_genome is not None:
                # Clone & mutate
                nn = neuralNet(input_size, hidden_sizes, output_size)
                mutated = self.mutate(np.copy(best_genome))
                nn.set_genome(mutated)
                population.append(nn)
            else:
                # Random genome
                nn = neuralNet(input_size, hidden_sizes, output_size)
                population.append(nn)
        return population
    
    def evaluate_population(self, fitness_scores):
        """Assign fitness scores to population."""
        if len(fitness_scores) != len(self.population):
            raise ValueError("Fitness scores must match population size")
        for i, score in enumerate(fitness_scores):
            if score is None:
                raise ValueError(f"Fitness score for individual {i} is None")
            validated_score = max(0.0, float(score))
            self.fitness_scores[i] = validated_score
            setattr(self.population[i], 'fitness_value', validated_score)
    
    def selection(self):
        """Tournament selection with increased pressure."""
        selected = []
        tournament_size = min(5, len(self.population))  # Increased from 3
        for _ in range(self.population_size):
            contestants = random.sample(list(enumerate(self.population)), tournament_size)
            winner_idx = max(contestants, key=lambda x: self.fitness_scores[x[0]])[0]
            selected.append(self.population[winner_idx])
        return selected
    
    def crossover(self, parent1, parent2):
        """Arithmetic crossover for real-valued genomes."""
        try:
            child = neuralNet(parent1.layer_sizes[0], 
                            parent1.layer_sizes[1:-1], 
                            parent1.layer_sizes[-1])
            genome1 = parent1.get_genome()
            genome2 = parent2.get_genome()
            if len(genome1) != len(genome2):
                raise ValueError("Parent genomes must be same length")
            # Arithmetic crossover
            alpha = random.random()
            child_genome = alpha * genome1 + (1 - alpha) * genome2
            child.set_genome(child_genome)
            return child
        except Exception as e:
            print(f"Error in crossover: {e}")
            return parent1 if random.random() < 0.5 else parent2
    
    def mutate(self, genome):
        """Mutate with smaller step size."""
        try:
            mask = np.random.rand(len(genome)) < self.mutation_rate
            noise = np.random.normal(0, 0.01, len(genome))  # Smaller noise
            mutated = genome + (mask * noise)
            return np.clip(mutated, -1.5, 1.5)
        except Exception as e:
            print(f"Error in mutation: {e}")
            return genome
    
    def create_next_generation(self):
        """Create new generation with diversity maintenance."""
        try:
            # Sort population by fitness
            sorted_indices = np.argsort(self.fitness_scores)[::-1]
            sorted_pop = [self.population[i] for i in sorted_indices]
            
            # Keep elites
            new_population = []
            for elite in sorted_pop[:self.elite_size]:
                clone = neuralNet(elite.layer_sizes[0], elite.layer_sizes[1:-1], elite.layer_sizes[-1])
                clone.set_genome(elite.get_genome().copy())
                new_population.append(clone)
            
            # Record best performer
            if len(sorted_pop) > 0:
                current_best_fitness = self.fitness_scores[sorted_indices[0]]
                self.best_fitness_history.append(current_best_fitness)
                self.best_genome_history.append(sorted_pop[0].get_genome().copy())
                
                # Check for stagnation
                if len(self.best_fitness_history) > 10:
                    recent_best = max(self.best_fitness_history[-10:])
                    if current_best_fitness <= recent_best:
                        self.stagnation_counter += 1
                    else:
                        self.stagnation_counter = 0
                
                # Increase mutation rate if stagnating
                original_mutation_rate = self.mutation_rate
                if self.stagnation_counter > 5:
                    self.mutation_rate = min(0.1, self.mutation_rate * 2)
            
            # Selection
            parents = self.selection()
            
            # Crossover and mutation
            while len(new_population) < self.population_size:
                try:
                    parent1, parent2 = random.sample(parents, 2)
                    child = self.crossover(parent1, parent2)
                    child_genome = child.get_genome()
                    mutated_genome = self.mutate(child_genome)
                    child.set_genome(mutated_genome)
                    new_population.append(child)
                except Exception as e:
                    print(f"Error creating child: {e}")
                    continue
            
            self.population = new_population
            self.generation += 1
            self.mutation_rate = original_mutation_rate  # Reset mutation rate
            
            # Save best genome periodically
            if self.generation % 2 == 0:
                self.save_best_genome()
                
        except Exception as e:
            print(f"Error in generation creation: {e}")
            self.population = self.initialize_population(24, [16, 8], 3)
            self.fitness_scores = [0.0] * self.population_size
    
    def save_best_genome(self):
        """Save all genomes with the best one at the top"""
        try:
            if not self.best_genome_history:
                return
                
            os.makedirs("result", exist_ok=True)
            filename = "result/best_genomes.csv"
            
            # Prepare header row
            header = ['generation', 'fitness'] + [f'gene_{i}' for i in range(len(self.best_genome_history[0]))]
            
            # Sort genomes by fitness (descending)
            fitness_genome_pairs = list(zip(self.best_fitness_history, self.best_genome_history))
            sorted_pairs = sorted(fitness_genome_pairs, key=lambda x: x[0], reverse=True)
            
            # Create rows for all genomes
            rows = [header]
            for i, (fitness, genome) in enumerate(sorted_pairs):
                gen_num = i  # You might want to track actual generation numbers
                rows.append([gen_num, fitness] + list(genome))
            
            # Write to file
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                
            print(f"Saved {len(sorted_pairs)} genomes to {filename}")
            
        except Exception as e:
            print(f"Error saving genomes: {e}")
            import traceback
            traceback.print_exc()
    
    def load_best_genome(self):
        """Load the best 20 genomes from CSV for a population of 100"""
        try:
            filename = "result/best_genomes.csv"
            if not os.path.exists(filename):
                return None
                
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = [row for row in reader if row]
                
            if not rows:
                return None
                
            # Parse rows and sort by fitness
            valid_rows = []
            for row in rows:
                try:
                    fitness = float(row[1])
                    genome = np.array([float(gene) for gene in row[2:]])
                    valid_rows.append((fitness, genome))
                except (ValueError, IndexError):
                    continue
                    
            if not valid_rows:
                return None
                
            # Sort by fitness (descending)
            valid_rows.sort(key=lambda x: x[0], reverse=True)
            
            # Take the top genome for return value
            best_genome = valid_rows[0][1]
            print(f"Loaded best genome with fitness {valid_rows[0][0]}")
            
            # Store the top 20 genomes for population initialization
            self.top_genomes = [genome for _, genome in valid_rows[:20]]
            print(f"Loaded top {len(self.top_genomes)} genomes for population initialization")
            
            return best_genome
            
        except Exception as e:
            print(f"Error loading genomes: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_population(self):
        return self.population
    def initialize_population_with_top_genomes(self, input_size, hidden_layers, output_size):
        if hasattr(self, 'top_genomes') and self.top_genomes:
            num_top = min(len(self.top_genomes), len(self.population))
            
            # Copy top genomes directly
            for i in range(num_top):
                self.population[i].set_genome(self.top_genomes[i])
            
            # Fill the rest with mutated top genomes + some randomness
            for i in range(num_top, len(self.population)):
                # Choose a random top genome to mutate
                parent_idx = np.random.randint(0, len(self.top_genomes))  # Random index
                parent_genome = self.top_genomes[parent_idx]  # Extract genome
                mutated_genome = self.mutate(parent_genome)  # Example mutation
                self.population[i].set_genome(mutated_genome)
                
                # Optional: Add randomness for a subset (e.g., 20% of the remaining)
                if np.random.rand() < 0.2:
                    self.population[i] = neuralNet(input_size, hidden_layers, output_size)
            
            print(f"Initialized with {num_top} top genomes and {len(self.population)-num_top} mutated/random")
        else:
            # Default random initialization
            for i in range(len(self.population)):
                self.population[i] = neuralNet(input_size, hidden_layers, output_size)
            print("Population initialized randomly")
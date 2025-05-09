import numpy as np
import random
import csv
import os
from neuralNet import neuralNet
import time
class Trainer:
    def __init__(self, population_size=100, mutation_rate=0.1, elite_size=10):
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
            # Using 8 inputs (5 base + 3 track sensors) and 1 hidden layer of 12 neurons
            nn = neuralNet(8, [12], 3)
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
            validated_score = float(score)
            self.fitness_scores[i] = validated_score
            setattr(self.population[i], 'fitness_value', validated_score)
    
    def selection(self):
        """Tournament selection with increased pressure."""
        selected = []
        tournament_size = min(20, len(self.population))  # Increased from 3
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
            print("Mutation rate:", self.mutation_rate)
            time.sleep(1)
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
            # Reset with the new architecture: 8 inputs, 1 hidden layer with 12 neurons, 3 outputs
            self.population = self.initialize_population(8, [12], 3)
            self.fitness_scores = [0.0] * self.population_size


    def save_best_genome(self):
        """Save only unique genomes to CSV, avoiding duplicates"""
        try:
            if not self.best_genome_history:
                return

            os.makedirs("result", exist_ok=True)
            filename = "result/NewBraking_2.csv"

            # Read existing genomes if file exists
            existing_genomes = set()
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) > 2:  # Ensure row has genes
                            # Convert genome part to tuple for hashability
                            genome_tuple = tuple(float(gene) for gene in row[2:])
                            existing_genomes.add(genome_tuple)

            # Prepare new genomes to save (unique only)
            new_genomes_to_save = []
            header = ['generation', 'fitness'] + [f'gene_{i}' for i in range(len(self.best_genome_history[0]))]
            
            # Check each genome in history for uniqueness
            for gen_num, (fitness, genome) in enumerate(zip(self.best_fitness_history, self.best_genome_history)):
                genome_tuple = tuple(genome)  # Convert to tuple for comparison
                if genome_tuple not in existing_genomes:
                    new_genomes_to_save.append([gen_num, fitness] + list(genome))
                    existing_genomes.add(genome_tuple)  # Mark as saved

            if not new_genomes_to_save:
                print("No new unique genomes to save")
                return

            # Write to file (append mode)
            write_header = not os.path.exists(filename)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerows(new_genomes_to_save)

            print(f"Saved {len(new_genomes_to_save)} new unique genomes to {filename}")

        except Exception as e:
            print(f"Error saving genomes: {e}")
            import traceback
            traceback.print_exc()

    
    def load_best_genome(self):
        """Load the top 20 unique genomes from CSV for a population of 100"""
        try:
            filename = "result/NewBraking_2.csv"
            if not os.path.exists(filename):
                return None
                
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                rows = [row for row in reader if row]  # Skip empty rows
                
            if not rows:
                return None
                
            # Parse rows, extract fitness and genome, and filter duplicates
            unique_genomes = {}
            for row in rows:
                try:
                    fitness = float(row[1])
                    genome = tuple(float(gene) for gene in row[2:])  # Convert to tuple (hashable)
                    
                    # Keep only the highest fitness version of each genome
                    if genome not in unique_genomes or fitness > unique_genomes[genome][0]:
                        unique_genomes[genome] = (fitness, np.array(genome))
                except (ValueError, IndexError):
                    continue
                    
            if not unique_genomes:
                return None
                
            # Sort genomes by fitness (descending)
            sorted_genomes = sorted(unique_genomes.values(), key=lambda x: x[0], reverse=True)
            
            # Take the best genome for return value
            best_genome = sorted_genomes[0][1]
            print(f"Loaded best unique genome with fitness {sorted_genomes[0][0]}")
            
            # Store top 20 unique genomes for population initialization
            self.top_genomes = [genome for _, genome in sorted_genomes[:20]]
            print(f"Loaded {len(self.top_genomes)} unique top genomes")
            
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
                
              #  Optional: Add randomness for a subset (e.g., 20% of the remaining)
                if np.random.rand() < 0.2:
                    self.population[i] = neuralNet(input_size, hidden_layers, output_size)
            
            print(f"Initialized with {num_top} top genomes and {len(self.population)-num_top} mutated/random")
        else:
            # Default random initialization
            for i in range(len(self.population)):
                self.population[i] = neuralNet(input_size, hidden_layers, output_size)
            print("Population initialized randomly")
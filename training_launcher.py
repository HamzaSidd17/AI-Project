import socket
import sys
import argparse
import multiprocessing
import time
import driver
import subprocess
import os
import traceback
from trainer import Trainer

def launch_torcs_instance(instance_id, base_port):
    vision_port = base_port + instance_id * 10
    control_port = vision_port + 1
    env = os.environ.copy()
    env['TORCS_PORT'] = str(vision_port)
    env['TORCS_CONTROL_PORT'] = str(control_port)
    torcs_dir = '/home/dani/torcs-1.3.7/BUILD/bin'
    torcs_executable = os.path.join(torcs_dir, 'torcs')
    try:
        process = subprocess.Popen(
            ['./torcs', '-p', str(vision_port)],
            env=env,
            cwd=torcs_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"[INFO] Launched TORCS instance {instance_id} on ports {vision_port} and {control_port}")
        time.sleep(5)  # Wait for TORCS to initialize
        return process, vision_port
    except Exception as e:
        print(f"[ERROR] Failed to launch TORCS instance {instance_id}: {e}")
        return None, None

def run_bot(instance_id, host_ip, vision_port, bot_id, max_episodes, max_steps, track, stage, nn):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
    except socket.error as e:
        print(f"[ERROR] Instance {instance_id}: Could not create socket: {e}")
        return 0.0
    
    shutdownClient = False
    curEpisode = 0
    d = driver.Driver(stage)
    if nn is not None:
        try:
            d.getNueralNetwork().set_genome(nn.get_genome())
        except Exception as e:
            print(f"[ERROR] Instance {instance_id}: Failed to set neural network genome: {e}")
            sock.close()
            return 0.0
    
    start_time = time.time()
    fitness_data = []
    lap_completed = False
    run = 0
    current_fitness = 0.0
    last_position = None
    last_movement_time = None
    race_started = False
    startBuffer = 7.0
    
    while not shutdownClient:
        # Connect to server
        while True:
            print(f"[BOT-{instance_id}] Sending id to server: {bot_id}")
            buf = bot_id + d.init()
            try:
                sock.sendto(buf.encode(), (host_ip, vision_port))
            except socket.error as e:
                print(f"[ERROR] Instance {instance_id}: Failed to send data: {e}. Exiting...")
                sock.close()
                return 0.0
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')
            except socket.error:
                print(f"[BOT-{instance_id}] No response yet... retrying...")
                continue
            if buf.find('***identified***') >= 0:
                print(f"[BOT-{instance_id}] Connected: {buf}")
                last_movement_time = time.time()
                race_started = True
                break
        
        currentStep = 0
        print(f"[BOT-{instance_id}] Entering driving loop")
        while True:
            buf = None
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')
            except socket.error:
                print(f"[BOT-{instance_id}] Waiting for data...")
                continue
            
            if buf and '***shutdown***' in buf:
                d.onShutDown()
                shutdownClient = True
                print(f"[BOT-{instance_id}] Shutdown signal received.")
                break
            elif buf and '***restart***' in buf:
                d.onRestart()
                print(f"[BOT-{instance_id}] Restart signal received.")
                break
            
            if buf:
                currentStep += 1
                if start_time is not None and time.time() - start_time >= 7.0:
                    startBuffer = 0.0
                print(f"[BOT-{instance_id}] Step {currentStep}: Processing data")
                
                if currentStep < max_steps:
                    try:
                        buf, output = d.drive(buf)
                        
                    except Exception as e:
                        print(f"[BOT-{instance_id}] Error in drive: {e}")
                        traceback.print_exc()
                        break
                    
                    try:
                        if d is None:
                            print(f"[BOT-{instance_id}] ERROR: Driver object is None!")
                            fitness_data.append(0.0)
                            break
                        if d.state is None:
                            print(f"[BOT-{instance_id}] ERROR: Driver state is None!")
                            fitness_data.append(0.0)
                            break
                        
                        # Check for lap completion
                        if d.state.lastLapTime is not None and d.state.lastLapTime > 0:
                            lap_completed = True
                            print(f"[BOT-{instance_id}] Lap completed with time: {d.state.lastLapTime}")
                        
                        # Crash detection
                        # if hasattr(d.state, 'damage') and d.state.damage > 50:
                        #     print(f"[BOT-{instance_id}] Crash detected. Ending episode.")
                        #     break
                        
                        # Stalling detection
                        # current_position = getattr(d.state, 'distRaced', 0.0)
                        # current_time = time.time()
                        # print(f"[BOT-{instance_id}] Current position: {current_position}, Last position: {last_position}")
                        # print(f"[BOT-{instance_id}] Start Buffer: {startBuffer} Start Time: {start_time} Current Time: {current_time}")
                        # if last_position is not None and race_started:
                        #     if current_time - last_movement_time + startBuffer >= 10.0:
                        #         print(f"[BOT-{instance_id}] Stalled for 10 seconds. Ending episode.")
                        #         if abs(current_position - last_position) < 1:
                        #             print(f"[BOT-{instance_id}] Stalled for 7 seconds. Ending episode.")
                        #             break
                        #     else:
                        #         last_movement_time = current_time
                        
                        # last_position = current_position
                        
                        # Fitness calculation
                        try:
                            neural_network = d.getNueralNetwork()
                            if neural_network is None:
                                print(f"[BOT-{instance_id}] ERROR: Neural network is None!")
                            elif neural_network is not None and hasattr(neural_network, 'fitness') and callable(getattr(neural_network, 'fitness')):
                                current_fitness += neural_network.fitness(d.state, output, lap_completed)
                                fitness_data.append(current_fitness)
                                run += 1
                                print(f"[BOT-{instance_id}] Fitness calculated: {current_fitness}")
                            else:
                                if hasattr(d.state, 'distRaced') and d.state.distRaced is not None:
                                    current_fitness += d.state.distRaced * (1.0 if lap_completed else 0.5)
                                else:
                                    current_fitness += 0.0
                                print(f"[BOT-{instance_id}] Using default fitness calculation: {current_fitness}")
                        except Exception as e:
                            print(f"[BOT-{instance_id}] Error calculating fitness: {e}")
                            traceback.print_exc()
                            fitness_data.append(0.0)

                        
                    except Exception as e:
                        print(f"[BOT-{instance_id}] Error processing data: {e}")
                        traceback.print_exc()
                        break
                    try:
                        sock.sendto(buf.encode(), (host_ip, vision_port))
                    except socket.error as e:
                        print(f"[BOT-{instance_id}] Failed to send data: {e}...Exiting...")
                        break
                else:
                    buf = '(meta 1)'
                    try:
                        sock.sendto(buf.encode(), (host_ip, vision_port))
                    except socket.error as e:
                        print(f"[BOT-{instance_id}] Failed to send meta command: {e}...Exiting...")
                        break
            
            elapsed_time = time.time() - start_time
            if elapsed_time >= 200 or lap_completed:
                print(f"[BOT-{instance_id}] {'Lap completed' if lap_completed else '200 seconds passed'}. Ending episode.")
                break
        
        # Send restart command to TORCS
        if not shutdownClient:
            print(f"[BOT-{instance_id}] Sending restart command to TORCS")
            try:
                sock.sendto('(meta 1)'.encode(), (host_ip, vision_port))
                d.onRestart()
                
                # Wait for restart confirmation
                try:
                    buf, addr = sock.recvfrom(1000)
                    buf = buf.decode('utf-8')
                    if '***restart***' in buf:
                        print(f"[BOT-{instance_id}] TORCS restarted successfully")
                    else:
                        print(f"[BOT-{instance_id}] Unexpected response after restart: {buf}")
                except socket.error:
                    print(f"[BOT-{instance_id}] No restart confirmation received, proceeding...")
            except socket.error as e:
                print(f"[BOT-{instance_id}] Failed to send restart command: {e}")
        
        final_fitness = current_fitness / run 
        if nn is not None:
            nn.fitness_value = final_fitness
            if hasattr(nn, '_index') and hasattr(nn, '_trainer'):
                nn._trainer.fitness_scores[nn._index] = final_fitness
        
        curEpisode += 1
        if curEpisode >= max_episodes:
            shutdownClient = True
        print(f"[BOT-{instance_id}] Episode {curEpisode} completed with fitness: {final_fitness:.2f}")
    
    
    sock.close()
    print(f"[BOT-{instance_id}] Finished with final fitness: {final_fitness:.2f}")
    return final_fitness

def run_bot_wrapper(instance_id, host_ip, port, bot_id, max_episodes, max_steps, track, stage, nn, results):
    fitness = run_bot(instance_id, host_ip, port, bot_id, max_episodes, max_steps, track, stage, nn)
    results[instance_id] = fitness

def run_generation(trainer, host_ip, base_port, max_episodes, max_steps, track, stage):
    population = trainer.get_population()
    all_fitness_scores = []
    batch_size = 10
    num_batches = (len(population) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(population))
        batch_population = population[start:end]

        processes = []
        manager = multiprocessing.Manager()
        results = manager.dict()

        # Start processes for each bot in the batch
        for j, nn in enumerate(batch_population):
            i = start + j
            p = multiprocessing.Process(
                target=run_bot_wrapper,
                args=(i, host_ip, base_port + j, f'SCR_{j}', 
                      max_episodes, max_steps, track, stage, nn, results)
            )
            processes.append(p)
            p.start()
            time.sleep(1)

        # Wait for all processes in the batch to finish
        for p in processes:
            p.join()

        # Collect fitness scores for this batch
        batch_fitness_scores = [results.get(start + k, 0.0) for k in range(len(processes))]
        all_fitness_scores.extend(batch_fitness_scores)

    # Evaluate all fitness scores after all batches
    trainer.evaluate_population(all_fitness_scores)
    trainer.create_next_generation()

def main():
    parser = argparse.ArgumentParser(description='TORCS Genetic Algorithm Trainer')
    parser.add_argument('--bots', type=int, default=100, help='Population size')
    parser.add_argument('--port', type=int, default=3001, help='Base port number')
    parser.add_argument('--host', type=str, default='localhost', help='Host IP')
    parser.add_argument('--episodes', type=int, default=1, help='Episodes per generation')
    parser.add_argument('--steps', type=int, default=500, help='Steps per episode')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations')
    parser.add_argument('--track', type=str, default=None, help='Track name')
    parser.add_argument('--stage', type=int, default=3, help='Stage (0-3)')
    args = parser.parse_args()
    trainer = Trainer(population_size=args.bots)
    torcs_process, vision_port = launch_torcs_instance(0, args.port)
    best_genomes = trainer.load_best_genome()
    if best_genomes is not None:
        print("Loaded best genome from previous run")
        trainer.initialize_population_with_top_genomes(24, [16, 8], 3)

    if torcs_process is None:
        print("[ERROR] Failed to launch TORCS. Exiting...")
        sys.exit(1)
    try:
        for gen in range(args.generations):
            print(f"\n=== Running Generation {gen + 1}/{args.generations} ===")
            run_generation(trainer, args.host, args.port, args.episodes, args.steps, args.track, args.stage)
            if trainer.fitness_scores:
                best_fitness = max(trainer.fitness_scores)
                avg_fitness = sum(trainer.fitness_scores) / len(trainer.fitness_scores)
                print(f"Generation {gen + 1}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}")
            else:
                print(f"Generation {gen + 1}: No fitness scores available")
        #trainer.save_best_genome()
        print("\nTraining complete. Best genome saved to best_genomes.csv")
    finally:
        torcs_process.terminate()
        torcs_process.wait()

if __name__ == '__main__':
    main()
import socket
import sys
import argparse
import multiprocessing
import time
import driver
import subprocess
import os

# python training_launcher.py --bots 2 --maxEpisodes 5 --maxSteps 3000 --stage 2


def launch_torcs_instance(instance_id, base_port):
    # Set unique ports for each instance
    vision_port = base_port + instance_id * 10
    control_port = vision_port + 1

    # Set environment variables for the instance
    env = os.environ.copy()
    env['TORCS_PORT'] = str(vision_port)
    env['TORCS_CONTROL_PORT'] = str(control_port)

    # Dynamically get the absolute path to the TORCS directory and executable
    # copy torcs into project folder or provide your path here
    current_dir = os.path.dirname(os.path.abspath(__file__))
    torcs_dir = os.path.join(current_dir, 'torcs')
    torcs_executable = os.path.join(torcs_dir, 'wtorcs.exe')

    # Set working directory (cwd) to the TORCS folder so it finds data/
    # subprocess.Popen([torcs_executable, '-p', str(vision_port)], env=env, cwd=torcs_dir)
    subprocess.Popen(f'start "" "{torcs_executable}" -p {vision_port}', shell=True, env=env, cwd=torcs_dir)

    print(f"[INFO] Launched TORCS instance {instance_id} on ports {vision_port} and {control_port}")
    time.sleep(5)
    return vision_port, control_port    

def run_bot(instance_id, host_ip, vision_port, bot_id, max_episodes, max_steps, track, stage):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
    except socket.error:
        print(f"[ERROR] Instance {instance_id}: Could not create socket.")
        return

    # curr_network = trainer.get_random_genome()

    shutdownClient = False
    curEpisode = 0
    verbose = False
    d = driver.Driver(stage)
    # d.set_neural_net(curr_network)

    while not shutdownClient:
        while True:
            print(f"[BOT-{instance_id}] Sending id to server: {bot_id}")
            buf = bot_id + d.init()

            try:
                sock.sendto(buf.encode(), (host_ip, vision_port))
            except socket.error:
                print(f"[ERROR] Instance {instance_id}: Failed to send data. Exiting...")
                return

            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')
            except socket.error:
                print(f"[BOT-{instance_id}] No response yet... retrying...")

            if buf.find('***identified***') >= 0:
                print(f"[BOT-{instance_id}] Connected: {buf}")
                break

        currentStep = 0
        while True:
            buf = None
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')
            except socket.error:
                print(f"[BOT-{instance_id}] Waiting for data...")

            if buf and '***shutdown***' in buf:
                d.onShutDown()
                shutdownClient = True
                print(f"[BOT-{instance_id}] Shutdown signal received.")
                break
            elif buf and '***restart***' in buf:
                # next_network = trainer.get_random_genome()
                # curr_network = next_network
                d.onRestart()
                print(f"[BOT-{instance_id}] Restart signal received.")
                break

            currentStep += 1
            if currentStep != max_steps:
                if buf:
                    buf = d.drive(buf)
                    print(f"[BOT-{instance_id}] Received: {buf}")
            else:
                # trainer.train(50)
                buf = '(meta 1)'

            if buf:
                try:
                    sock.sendto(buf.encode(), (host_ip, vision_port))
                except socket.error:
                    print(f"[BOT-{instance_id}] Failed to send data...Exiting...")
                    sys.exit(-1)

        curEpisode += 1
        if curEpisode == max_episodes:
            shutdownClient = True

    sock.close()
    print(f"[BOT-{instance_id}] Finished.")

def launch_bots_only(num_bots, host_ip, port, max_episodes, max_steps, track, stage):
    processes = []

    _port = port
    for i in range(num_bots):
        bot_id = f'SCR_{i + 1}'
        p = multiprocessing.Process(
            target=run_bot,
            args=(i, host_ip, _port, bot_id, max_episodes, max_steps, track, stage)
        )
        _port += 1
        processes.append(p)
        p.start()
        time.sleep(1)  # Small delay between bot launches

    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple TORCS bots against one GUI instance.')
    parser.add_argument('--bots', type=int, default=4, help='Number of bots to run.')
    parser.add_argument('--port', type=int, default=3001, help='Port number TORCS is using.')
    parser.add_argument('--host', type=str, default='localhost', help='Host IP address.')
    parser.add_argument('--maxEpisodes', type=int, default=1, help='Max learning episodes.')
    parser.add_argument('--maxSteps', type=int, default=500, help='Max steps per episode.')
    parser.add_argument('--track', type=str, default=None, help='Track name.')
    parser.add_argument('--stage', type=int, default=3, help='Stage: 0-WarmUp, 1-Qualifying, 2-Race, 3-Unknown.')
    args = parser.parse_args()

    launch_torcs_instance(0, args.port)

    launch_bots_only(
        num_bots=args.bots,
        host_ip=args.host,
        port=args.port,
        max_episodes=args.maxEpisodes,
        max_steps=args.maxSteps,
        track=args.track,
        stage=args.stage
    )


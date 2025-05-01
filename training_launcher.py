import subprocess
import time
import os
import socket
import sys
import argparse
import multiprocessing
import driver

def launch_torcs_instance(instance_id, base_port):
    # Set unique ports for each instance
    vision_port = base_port + instance_id * 10
    control_port = vision_port + 1

    # Set environment variables for the instance
    env = os.environ.copy()
    env['TORCS_PORT'] = str(vision_port)
    env['TORCS_CONTROL_PORT'] = str(control_port)

    # Specify the full path to the TORCS executable
    torcs_executable = r'D:\torcs\wtorcs.exe'

    # Launch the TORCS instance with visible GUI
    subprocess.Popen([torcs_executable, '-p', str(vision_port)], env=env)
    print(f"[INFO] Launched TORCS instance {instance_id} on ports {vision_port} and {control_port}")
    time.sleep(5)  # Wait for the TORCS GUI to fully load
    return vision_port, control_port


def run_bot(instance_id, host_ip, vision_port, bot_id, max_episodes, max_steps, track, stage):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
    except socket.error as msg:
        print(f"[ERROR] Instance {instance_id}: Could not create socket.")
        return

    shutdownClient = False
    curEpisode = 0
    verbose = False
    d = driver.Driver(stage)

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
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')
            except socket.error:
                print(f"[BOT-{instance_id}] Waiting for data...")

            if verbose:
                print(f"[BOT-{instance_id}] Received: {buf}")

            if buf and '***shutdown***' in buf:
                d.onShutDown()
                shutdownClient = True
                print(f"[BOT-{instance_id}] Shutdown signal received.")
                break
            elif buf and '***restart***' in buf:
                d.onRestart()
                print(f"[BOT-{instance_id}] Restart signal received.")
                break

            currentStep += 1
            if currentStep != max_steps:
                if buf:
                    buf = d.drive(buf)
                    print(f"[BOT-{instance_id}] Driving data: {buf}")
            else:
                buf = '(meta 1)'

            if verbose:
                print(f"[BOT-{instance_id}] Sending: {buf}")

            if buf:
                try:
                    sock.sendto(buf.encode(), (host_ip, vision_port))
                except socket.error:
                    print(f"[ERROR] Instance {instance_id}: Failed to send data. Exiting...")
                    return

        curEpisode += 1
        if curEpisode == max_episodes:
            shutdownClient = True

    sock.close()
    print(f"[BOT-{instance_id}] Finished.")

def launch_all_instances(num_instances, base_port, host_ip, max_episodes, max_steps, track, stage):
    processes = []

    for i in range(num_instances):
        vision_port, _ = launch_torcs_instance(i, base_port)
        bot_id = f'SCR_{i}'

        # Launch bot as separate process
        p = multiprocessing.Process(
            target=run_bot,
            args=(i, host_ip, vision_port, bot_id, max_episodes, max_steps, track, stage)
        )
        processes.append(p)
        p.start()
        time.sleep(2)  # Stagger launch to reduce race conditions

    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch multiple TORCS instances and bots.')
    parser.add_argument('--instances', type=int, default=4, help='Number of TORCS instances to launch.')
    parser.add_argument('--basePort', type=int, default=3001, help='Base port number.')
    parser.add_argument('--host', type=str, default='localhost', help='Host IP address.')
    parser.add_argument('--maxEpisodes', type=int, default=1, help='Max learning episodes.')
    parser.add_argument('--maxSteps', type=int, default=0, help='Max steps per episode.')
    parser.add_argument('--track', type=str, default=None, help='Track name.')
    parser.add_argument('--stage', type=int, default=3, help='Stage: 0-WarmUp, 1-Qualifying, 2-Race, 3-Unknown.')
    args = parser.parse_args()

    launch_all_instances(
        num_instances=args.instances,
        base_port=args.basePort,
        host_ip=args.host,
        max_episodes=args.maxEpisodes,
        max_steps=args.maxSteps,
        track=args.track,
        stage=args.stage
    )

import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect dataset"
    )

    parser.add_argument("--carla_garage_root", type=str, required=True,
                        help="Path to carla_garage")
    parser.add_argument("--script_path", type=str, required=True,
                        help="Main script pth to run the simulation")
    parser.add_argument("--common_library_root", type=str, required=True,
                        help="Path to common_library")
    parser.add_argument("--scenario_path", type=str, required=True,
                        help="Path to scenario root folder")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the dataset")
    parser.add_argument("--agent", type=str, required=True,
                        help="Path to agent file used for data collection")
    parser.add_argument("--host", type=str, required=True,
                        help="IP addres of the machine where CARLA server is running. ")
    parser.add_argument("--port", type=str, required=True,
                        help="port num used to connect carla server")
    parser.add_argument("--trafficmanagerport", type=str, required=True,
                        help="port num for trafficmanager")
    parser.add_argument("--num_scenes", type=str, required=True,
                        help="Number of scenes(xml files) executed for each scenario(e.g. AccidentTwoWays, BlockedIntersection, ...). ")
    
    return parser.parse_args()

def main():
    args = parse_args()
#-----------------------------Configuration parameters------------------------------
    carla_garage_root = args.carla_garage_root
    script_path = args.script_path

    if args.num_scenes != "None":
        num_scenes = int(args.num_scenes)
        print(f"Try to collect {num_scenes} scenes for each scenarios respectively")
    else:
        num_scenes = None #Number of scenes(xml files) executed for each scenario(e.g. AccidentTwoWays, BlockedIntersection, ...). "
        print(f"Try to collect all scenes involved in each scenarios")
    
    scenario_path = args.scenario_path
    repetitions = "1" #Repetitions for each scenes
    trafficmanagerseed = "0"
    
    save_path = args.save_path
    host = args.host

    #Fixed parameter
    track = "MAP_QUALIFIER"
    agent = args.agent
    debug = "0"
    resume = "1"
    timeout = "2000"
    port = args.port
    trafficmanagerport = args.trafficmanagerport
#-----------------------------------------------------------------------------------

    env = os.environ.copy()
    env.update({
                "REPETITION": f"{repetitions}",
                "DEBUG_CHALLENGE": f"{debug}",
                "PTH_ROUTE": "",
                "TEAM_AGENT": f"{agent}",
                "CHALLENGE_TRACK_CODENAME": f"{track}",
                "ROUTES": "",
                "PORT": f"{port}",
                "TM_PORT": f"{trafficmanagerport}",
                "TOWN": "",
                "CHECKPOINT_ENDPOINT": "",
                "TEAM_CONFIG": "",
                "PTH_LOG": f"{save_path}",
                "RESUME": f"{resume}",
                "DATAGEN": "1",
                "SAVE_PATH": f"{save_path}",
                "TM_SEED": f"{trafficmanagerseed}",
                "CUDA_VISIBLE_DEVICES": "0"
            })

    scenarios = glob.glob(scenario_path, recursive=True)

    for scenario in scenarios:
        routes = glob.glob(f"{scenario}/*.xml")

        num_route = 0
        for route in routes:
            if num_scenes is not None and num_scenes < num_route:
                break
            else:
                path_routes = route.removesuffix(".xml")
                checkpoint = f"{path_routes}.json"
                agentconfig = f"{route}"

                town = re.search('Town(\\d+)', route).group(0)

                env["PTH_ROUTE"] = f"{path_routes}"
                env["ROUTES"] = f"{route}"
                env["TOWN"] = f"{town}"
                env["CHECKPOINT_ENDPOINT"] = f"{checkpoint}"
                env["TEAM_CONFIG"] = f"{agentconfig}"

                if os.path.exists(checkpoint):
                    print(f"skip scenario : {checkpoint}")
                    continue
                
                args = [
                    f"--host={host}",
                    f"--port={port}",
                    f"--traffic-manager-port={trafficmanagerport}",
                    f"--routes={route}",
                    f"--repetitions={repetitions}",
                    f"--track={track}",
                    f"--checkpoint={checkpoint}",
                    f"--agent={agent}",
                    f"--agent-config={agentconfig}",
                    f"--debug={debug}",
                    f"--resume={resume}",
                    f"--timeout={timeout}",
                    f"--traffic-manager-seed={trafficmanagerseed}"
                ]

                subprocess.run([sys.executable, script_path] + args, cwd=carla_garage_root, env=env)
                time.sleep(60)
                num_route += 1




if __name__ == "__main__":
    main()
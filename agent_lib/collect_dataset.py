import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
#-----------------------------Configuration parameters------------------------------
    root = Path(__file__).resolve().parent.parent.parent
    carla_garage_root = (root/"carla_garage").as_posix()
    script_path = f"{carla_garage_root}/leaderboard/leaderboard/leaderboard_evaluator_local.py"#main script to run the simulation
    common_library_root = (root/"common_library").as_posix()

    #Tunable parameter
    num_scenes = None #Number of scenes(xml files) executed for each scenario(e.g. AccidentTwoWays, BlockedIntersection, ...). 
    scenario_path =  f"{carla_garage_root}/data/*50x38_Town12*/*/"
    repetitions = "1" #Repetitions for each scenes
    trafficmanagerseed = "0"
    save_path = (root.parent/"logs/dataset/scenario").as_posix() #Dataset will be stored here. 
    host = "172.27.160.1" #IP addres of the machine where CARLA server is running. 

    #Fixed parameter
    track = "MAP_QUALIFIER"
    agent = f"{common_library_root}/agent_lib/data_agent.py" #Agent used for data collection. 
    debug = "0"
    resume = "1"
    timeout = "2000"
    port = "2000"
    trafficmanagerport = "2003"
#-----------------------------------------------------------------------------------

    env = os.environ.copy()
    env.update({
                "CARLA_ROOT": f"{carla_garage_root}/carla",
                "WORK_DIR": f"{carla_garage_root}",
                "SCENARIO_RUNNER_ROOT": f"{carla_garage_root}/scenario_runner_autopilot",
                "LEADERBOARD_ROOT": f"{carla_garage_root}/leaderboard_autopilot",
                "PYTHONPATH": os.environ.get("PYTHONPATH", "") +
                    f":{carla_garage_root}/carla/PythonAPI/carla" +
                    f":{carla_garage_root}/scenario_runner_autopilot" +
                    f":{carla_garage_root}/leaderboard_autopilot" +
                    f":{carla_garage_root}/team_code" +
                    f":{common_library_root}",
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
                    #print(f"Delete file : {checkpoint}")
                    #os.remove(checkpoint)
                
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
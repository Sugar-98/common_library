import glob
import os
import re
import subprocess
import sys
import time


def main():
#-----------------------------Configuration parameters------------------------------
    script_path = "/home/workspace/carla_garage/leaderboard/leaderboard/leaderboard_evaluator_local.py"#main script to run the simulation
    working_directory = "/home/workspace/carla_garage"

    #Tunable parameter
    repetitions_for_scenario = None
    data_root = "/home/workspace/carla_garage/data"
    scenario_path = f"{data_root}/*lb1_split_copy*/*/"
    repetitions = "1"
    trafficmanagerseed = "0"
    save_path = "/home/workspace/logs/dataset_dist/scenario"
    host = "172.27.160.1"
    trafficmanagerport = "2003"

    #Fixed parameter
    track = "MAP_QUALIFIER"
    agent = "/home/workspace/common_library/agent_lib/data_agent.py"
    debug = "0"
    resume = "1"
    timeout = "2000"
    port = "2000"
#-----------------------------------------------------------------------------------

    env = os.environ.copy()
    env.update({
                "CARLA_ROOT": "/home/workspace/carla_garage/carla",
                "WORK_DIR": f"{working_directory}",
                "SCENARIO_RUNNER_ROOT": "/home/workspace/carla_garage/scenario_runner_autopilot",
                "LEADERBOARD_ROOT": "/home/workspace/carla_garage/leaderboard_autopilot",
                "PYTHONPATH": os.environ.get("PYTHONPATH", "") +
                    ":/home/workspace/carla_garage/carla/PythonAPI/carla" +
                    ":/home/workspace/carla_garage/scenario_runner_autopilot" +
                    ":/home/workspace/carla_garage/leaderboard_autopilot" +
                    ":/home/workspace/carla_garage/team_code" +
                    ":/home/workspace/common_library",
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
            if repetitions_for_scenario is not None and repetitions_for_scenario < num_route:
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

                subprocess.run([sys.executable, script_path] + args, cwd=working_directory, env=env)
                time.sleep(60)
                num_route += 1




if __name__ == "__main__":
    main()
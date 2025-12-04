#!/bin/bash

# ==== Load environment settings ====
source ../../env.sh

# ==== Path to the Python script
MAIN_SCRIPT_PATH="$COMMON_LIBRARY_ROOT/agent_lib/collect_dataset.py"

# ==== Required arguments for the Python script ====
SCRIPT_PATH="$CARLA_GARAGE_ROOT/leaderboard/leaderboard/leaderboard_evaluator_local.py"
SCENARIO_ROOT="$CARLA_GARAGE_ROOT/data/*/*/"
SAVE_PATH="$PROJECT_ROOT/logs/dataset/scenario"
AGENT="$COMMON_LIBRARY_ROOT/agent_lib/data_agent.py"

# ==== Carla server info ====
HOST="172.27.160.1"
PORT="2000"
TM_PORT="2003"

# ==== Settings for data collection ====
NUM_SCENES="None" #Number of scenes(xml files) executed for each scenario(e.g. AccidentTwoWays, BlockedIntersection, ...). 
                  #All scenes will be executed if "None". 


# ==== Run ====
python3 "$MAIN_SCRIPT_PATH" \
    --carla_garage_root "$CARLA_GARAGE_ROOT" \
    --script_path "$SCRIPT_PATH" \
    --common_library_root "$COMMON_LIBRARY_ROOT" \
    --scenario_path "$SCENARIO_ROOT" \
    --save_path "$SAVE_PATH" \
    --agent "$AGENT" \
    --host "$HOST" \
    --port "$PORT" \
    --trafficmanagerport "$TM_PORT" \
    --num_scenes "$NUM_SCENES"
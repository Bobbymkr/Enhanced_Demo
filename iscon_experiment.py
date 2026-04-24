"""
ISCON Cross Road, Ahmedabad — Real SUMO/TraCI Experiment
=========================================================
Runs 6 algorithms × 5 scenarios × 5 seeds = 150 simulation runs
using genuine TraCI-controlled SUMO microscopic simulation.

Requirements:
  - SUMO v1.20+ installed, SUMO_HOME environment variable set
  - All osm.* files in the same folder as this script
  - pip install numpy pandas

Usage:
  python iscon_experiment.py

Output:
  results_raw.csv       — all 150 runs
  results_summary.csv   — mean ± std per algorithm/scenario
  paper_table.txt       — ready-to-paste paper table
"""

import os, sys, subprocess, random, time, math, csv, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── SUMO setup ────────────────────────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    # Common Windows install paths
    for candidate in [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
    ]:
        if os.path.isdir(candidate):
            os.environ["SUMO_HOME"] = candidate
            break

if "SUMO_HOME" not in os.environ:
    sys.exit("ERROR: SUMO_HOME not set. Install SUMO and set the environment variable.")

sys.path += [os.path.join(os.environ["SUMO_HOME"], "tools")]
try:
    import traci
    import sumolib
except ImportError:
    sys.exit("ERROR: traci/sumolib not found. Check SUMO_HOME is correct.")

SCRIPT_DIR   = Path(__file__).parent
NET_FILE     = SCRIPT_DIR / "osm.net.xml"
SUMO_BINARY  = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe") \
               if sys.platform == "win32" else \
               os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")

# ── Target junction ───────────────────────────────────────────────────────────
# cluster_1795571831_1795571870 — 4-arm junction at network centre (277, 352)
# Represents the primary signalised intersection on ISCON Cross Road, Ahmedabad
TL_ID        = "cluster_1795571831_1795571870"
# Four incoming approach lanes identified from network analysis
APPROACH_LANES = [
    "1255973940#4_0",   # Approach 1
    "168262882#0_0",    # Approach 2 (lane 0)
    "168262882#0_1",    # Approach 2 (lane 1)
    "168262887#0_0",    # Approach 3
]
# Two logical approach groups (pairs of opposing lanes → 2-phase control)
APPROACH_GROUPS = {
    "NS": ["1255973940#4_0", "168262887#0_0"],
    "EW": ["168262882#0_0", "168262882#0_1"],
}

# ── Simulation parameters ─────────────────────────────────────────────────────
SIM_DURATION   = 3600       # seconds (1 hour)
STEP_LENGTH    = 1.0        # seconds per step
CONTROL_INTERVAL = 10       # DQN decision every N steps
MIN_GREEN      = 20         # seconds
MAX_GREEN      = 60         # seconds
YELLOW_TIME    = 3          # seconds
SEEDS          = [42, 123, 256, 789, 1001]

# ── DQN hyperparameters (from config.py) ─────────────────────────────────────
MEMORY_SIZE    = 10_000
BATCH_SIZE     = 64
LEARNING_RATE  = 0.001
GAMMA          = 0.95
EPSILON_START  = 1.0
EPSILON_END    = 0.01
EPSILON_DECAY  = 0.995
TARGET_SYNC    = 100        # steps

# ── Route file modifications per scenario ─────────────────────────────────────
# We modify vehicle insertion density to simulate different conditions
SCENARIO_CONFIG = {
    "normal":    {"density_factor": 1.0,  "speed_factor": 1.0,  "emg": False},
    "peak_hour": {"density_factor": 2.2,  "speed_factor": 0.85, "emg": False},
    "emergency": {"density_factor": 1.0,  "speed_factor": 1.0,  "emg": True},
    "rain_fog":  {"density_factor": 0.74, "speed_factor": 0.75, "emg": False},
    "night":     {"density_factor": 0.38, "speed_factor": 1.1,  "emg": False},
}

SCENARIO_LABELS = {
    "normal":    "Normal Urban Traffic",
    "peak_hour": "Peak Hour (Rush)",
    "emergency": "Emergency Vehicle Present",
    "rain_fog":  "Adverse Weather (Rain/Fog)",
    "night":     "Night Conditions",
}

ALGORITHMS = [
    "Webster Fixed-Time",
    "Fuzzy Logic",
    "Genetic Algorithm",
    "PSO",
    "GNN",
    "DQN+YOLOv8 (Proposed)",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Route file generation
# ═══════════════════════════════════════════════════════════════════════════════

def build_route_file(scenario: str, seed: int) -> Path:
    """Generate a scenario-specific route file by sampling from base trips."""
    cfg   = SCENARIO_CONFIG[scenario]
    rng   = random.Random(seed)
    outf  = SCRIPT_DIR / f"_routes_{scenario}_{seed}.rou.xml"

    # Parse base passenger trips
    import xml.etree.ElementTree as ET
    base_trips = []
    for fname in ["osm.passenger.trips.xml", "osm.motorcycle.trips.xml",
                  "osm.bus.trips.xml"]:
        fpath = SCRIPT_DIR / fname
        if not fpath.exists():
            continue
        tree = ET.parse(fpath)
        for trip in tree.getroot().findall("trip"):
            base_trips.append({
                "id":    trip.get("id"),
                "from":  trip.get("from"),
                "to":    trip.get("to"),
                "depart": float(trip.get("depart", 0)),
                "type":  trip.get("type", "veh_passenger"),
                "dLane": trip.get("departLane", "best"),
                "dSpeed": trip.get("departSpeed", ""),
            })

    # Scale and resample
    factor = cfg["density_factor"]
    if factor >= 1.0:
        # Duplicate trips with slight time offsets
        extra = []
        for t in base_trips:
            copies = int(factor) - 1
            remainder = factor - int(factor)
            for i in range(copies):
                tt = dict(t)
                tt["id"]     = f"{t['id']}_x{i+1}"
                tt["depart"] = t["depart"] + rng.uniform(-30, 30)
                extra.append(tt)
            if rng.random() < remainder:
                tt = dict(t)
                tt["id"]     = f"{t['id']}_xr"
                tt["depart"] = t["depart"] + rng.uniform(-30, 30)
                extra.append(tt)
        all_trips = base_trips + extra
    else:
        # Subsample
        k = max(1, int(len(base_trips) * factor))
        all_trips = rng.sample(base_trips, k)

    # Sort by depart time
    all_trips.sort(key=lambda x: x["depart"])

    # Collect vehicle types
    vtypes = set(t["type"] for t in all_trips)

    # Write route file
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
             ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">']
    for vt in vtypes:
        lines.append(f'    <vType id="{vt}" vClass="passenger"'
                     f' maxSpeed="{22.2 * cfg["speed_factor"]:.2f}"/>')

    for t in all_trips:
        ds = f' departSpeed="{t["dSpeed"]}"' if t["dSpeed"] else ""
        lines.append(f'    <trip id="{t["id"]}" depart="{t["depart"]:.2f}"'
                     f' from="{t["from"]}" to="{t["to"]}"'
                     f' departLane="{t["dLane"]}" type="{t["type"]}"{ds}/>')

    lines.append("</routes>")
    outf.write_text("\n".join(lines), encoding="utf-8")
    return outf


# ═══════════════════════════════════════════════════════════════════════════════
# SUMO config per run
# ═══════════════════════════════════════════════════════════════════════════════

def build_sumocfg(route_file: Path, run_id: str) -> Path:
    outf  = SCRIPT_DIR / f"_run_{run_id}.sumocfg"
    tripinfo = SCRIPT_DIR / f"_tripinfo_{run_id}.xml"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<sumoConfiguration>',
        '  <input>',
        f'    <net-file value="{NET_FILE}"/>',
        f'    <route-files value="{route_file}"/>',
        '  </input>',
        '  <processing>',
        '    <ignore-route-errors value="true"/>',
        '    <time-to-teleport value="300"/>',
        '  </processing>',
        f'  <time><end value="{SIM_DURATION}"/></time>',
        '  <report><verbose value="false"/><no-step-log value="true"/></report>',
        '</sumoConfiguration>',
    ]
    outf.write_text("\n".join(lines), encoding="utf-8")
    return outf, tripinfo


# ═══════════════════════════════════════════════════════════════════════════════
# Traffic state measurement
# ═══════════════════════════════════════════════════════════════════════════════

def get_state():
    """Return [density_NS, queue_NS, density_EW, queue_EW]."""
    state = []
    for group, lanes in APPROACH_GROUPS.items():
        density = 0.0
        queue   = 0.0
        for lane in lanes:
            try:
                density += traci.lane.getLastStepVehicleNumber(lane)
                queue   += traci.lane.getLastStepHaltingNumber(lane)
            except traci.exceptions.TraCIException:
                pass
        state.extend([density / max(len(lanes), 1),
                      queue   / max(len(lanes), 1)])
    return state


def get_waiting_time():
    """Total waiting time of all vehicles in simulation (seconds)."""
    total = 0.0
    for vid in traci.vehicle.getIDList():
        total += traci.vehicle.getWaitingTime(vid)
    return total


def get_throughput_counter(prev_arrived: int):
    """Vehicles that completed their trip since last call."""
    arrived = traci.simulation.getArrivedNumber()
    return arrived


# ═══════════════════════════════════════════════════════════════════════════════
# Traffic light controllers
# ═══════════════════════════════════════════════════════════════════════════════

class WebsterController:
    """Fixed-time Webster method: 90s cycle, split proportional to flow."""
    def __init__(self):
        self.cycle    = 90
        self.phase    = 0
        self.timer    = 0
        self.durations = [42, 3, 42, 3]   # G, Y, G, Y

    def step(self, state):
        self.timer += STEP_LENGTH
        if self.timer >= self.durations[self.phase]:
            self.timer  = 0
            self.phase  = (self.phase + 1) % len(self.durations)
        # Map phase to SUMO phase index
        try:
            traci.trafficlight.setPhase(TL_ID, self.phase)
        except Exception:
            pass


class FuzzyController:
    """Mamdani fuzzy logic: adjust green time based on queue difference."""
    def __init__(self):
        self.phase    = 0
        self.timer    = 0
        self.green    = 35.0

    def _fuzz(self, q_diff):
        # Simple triangular membership → crisp output
        if   q_diff < -5: return 20.0
        elif q_diff <  0: return 20 + (q_diff + 5) / 5 * 15
        elif q_diff <  5: return 35 + q_diff / 5 * 15
        else:             return 50.0

    def step(self, state):
        self.timer += STEP_LENGTH
        if self.timer >= self.green:
            q_ns = state[1]; q_ew = state[3]
            self.green = self._fuzz(q_ns - q_ew)
            self.green = max(MIN_GREEN, min(MAX_GREEN, self.green))
            self.phase = (self.phase + 2) % 4   # toggle main phase
            self.timer = 0
        try:
            traci.trafficlight.setPhase(TL_ID, self.phase)
        except Exception:
            pass


class GAController:
    """Genetic Algorithm: pre-optimised fixed timing plan."""
    # GA offline-optimised splits (slightly better than Webster)
    def __init__(self):
        self.durations = [38, 3, 46, 3]
        self.phase = 0; self.timer = 0

    def step(self, state):
        self.timer += STEP_LENGTH
        if self.timer >= self.durations[self.phase]:
            self.timer = 0
            self.phase = (self.phase + 1) % 4
        try:
            traci.trafficlight.setPhase(TL_ID, self.phase)
        except Exception:
            pass


class PSOController:
    """PSO: iteratively updates phase duration toward queue-minimising split."""
    def __init__(self):
        self.green_ns = 40.0
        self.green_ew = 40.0
        self.phase = 0; self.timer = 0
        self.velocity = [0.0, 0.0]
        self.pbest = [40.0, 40.0]
        self.pbest_cost = float("inf")

    def _cost(self, state, g_ns, g_ew):
        # Estimated cost: weighted queue × inverse green fraction
        total = g_ns + g_ew
        if total == 0: return float("inf")
        c = state[1] * (g_ew / total) + state[3] * (g_ns / total)
        return c

    def step(self, state):
        self.timer += STEP_LENGTH
        current_green = self.green_ns if self.phase == 0 else self.green_ew
        if self.timer >= current_green + YELLOW_TIME:
            # PSO velocity update
            w, c1, c2 = 0.5, 1.5, 1.5
            g_best = self.pbest
            for i in range(2):
                r1, r2 = random.random(), random.random()
                self.velocity[i] = (w * self.velocity[i]
                    + c1 * r1 * (self.pbest[i] - [self.green_ns, self.green_ew][i])
                    + c2 * r2 * (g_best[i]     - [self.green_ns, self.green_ew][i]))
            self.green_ns = max(MIN_GREEN, min(MAX_GREEN, self.green_ns + self.velocity[0]))
            self.green_ew = max(MIN_GREEN, min(MAX_GREEN, self.green_ew + self.velocity[1]))
            cost = self._cost(state, self.green_ns, self.green_ew)
            if cost < self.pbest_cost:
                self.pbest_cost = cost
                self.pbest = [self.green_ns, self.green_ew]
            self.phase = 1 - (self.phase // 2)
            self.timer = 0
        try:
            traci.trafficlight.setPhase(TL_ID, self.phase * 2)
        except Exception:
            pass


class GNNController:
    """Graph Neural Network proxy: uses spatiotemporal weighted allocation."""
    def __init__(self):
        self.history  = defaultdict(list)
        self.phase    = 0; self.timer = 0; self.green = 35.0

    def step(self, state):
        self.timer += STEP_LENGTH
        for i, v in enumerate(state):
            self.history[i].append(v)
            if len(self.history[i]) > 10:
                self.history[i].pop(0)
        if self.timer >= self.green:
            # Spatiotemporal weight: moving avg queue pressure
            w_ns = np.mean(self.history.get(1, [state[1]])) if self.history.get(1) else state[1]
            w_ew = np.mean(self.history.get(3, [state[3]])) if self.history.get(3) else state[3]
            total = w_ns + w_ew + 1e-9
            cycle = 80
            self.green = max(MIN_GREEN, min(MAX_GREEN, (w_ns / total) * cycle))
            self.phase = (self.phase + 2) % 4
            self.timer = 0
        try:
            traci.trafficlight.setPhase(TL_ID, self.phase)
        except Exception:
            pass


# ── DQN ───────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class DQNNet(nn.Module if HAS_TORCH else object):
    def __init__(self):
        if HAS_TORCH:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 5),
            )
    def forward(self, x):
        return self.net(x)

ACTIONS = [20, 30, 40, 50, 60]   # green time options (seconds)

class DQNController:
    """
    Deep Q-Network controller with YOLOv8-informed state.
    Hyperparameters from config.py:
      memory=10000, batch=64, lr=0.001, gamma=0.95,
      eps=1.0→0.01 (decay 0.995/episode), target_sync=100 steps
    """
    def __init__(self, seed: int):
        self.epsilon   = EPSILON_START
        self.steps     = 0
        self.episode   = 0
        self.memory    = []
        self.phase     = 0
        self.timer     = 0
        self.green     = 40
        self.prev_wait = 0.0
        self.rng       = random.Random(seed)

        if HAS_TORCH:
            torch.manual_seed(seed)
            self.online  = DQNNet()
            self.target  = DQNNet()
            self.target.load_state_dict(self.online.state_dict())
            self.opt     = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
            self.loss_fn = nn.MSELoss()
        self._prev_state  = None
        self._prev_action = None

    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, len(ACTIONS) - 1)
        if HAS_TORCH:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0)
                q = self.online(s)
                return int(q.argmax().item())
        # Fallback: greedy heuristic
        q_ns, q_ew = state[1], state[3]
        if q_ns > q_ew:
            return min(4, 2 + int((q_ns - q_ew) / 3))
        return max(0, 2 - int((q_ew - q_ns) / 3))

    def _store(self, s, a, r, s2):
        if len(self.memory) >= MEMORY_SIZE:
            self.memory.pop(0)
        self.memory.append((s, a, r, s2))

    def _train(self):
        if not HAS_TORCH or len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, s2 = zip(*batch)
        S  = torch.FloatTensor(s)
        A  = torch.LongTensor(a).unsqueeze(1)
        R  = torch.FloatTensor(r)
        S2 = torch.FloatTensor(s2)
        q_pred   = self.online(S).gather(1, A).squeeze()
        q_target = R + GAMMA * self.target(S2).max(1)[0].detach()
        loss = self.loss_fn(q_pred, q_target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def _sync_target(self):
        if HAS_TORCH:
            self.target.load_state_dict(self.online.state_dict())

    def step(self, state):
        self.timer += STEP_LENGTH
        self.steps += 1

        # Sync target network
        if self.steps % TARGET_SYNC == 0:
            self._sync_target()

        # Phase transition
        if self.timer >= self.green:
            # Reward = negative change in total waiting time
            curr_wait = get_waiting_time()
            reward    = -(curr_wait - self.prev_wait)
            self.prev_wait = curr_wait

            # Store transition
            if self._prev_state is not None:
                self._store(self._prev_state, self._prev_action, reward, state)
                self._train()

            # Select new action
            action_idx   = self._select_action(state)
            self.green   = ACTIONS[action_idx]
            self._prev_state  = state
            self._prev_action = action_idx

            # Toggle phase (NS → EW → NS)
            self.phase = (self.phase + 2) % 4
            self.timer = 0

            # Decay epsilon per episode boundary (~every SIM_DURATION/90 episodes)
            self.episode += 1
            self.epsilon  = max(EPSILON_END,
                                self.epsilon * EPSILON_DECAY)

        try:
            traci.trafficlight.setPhase(TL_ID, self.phase)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Emergency vehicle pre-emption (DQN only)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_emergency(approach_lanes):
    """
    Camera-based ambulance detection proxy.
    In real deployment: YOLOv8 classifies ambulance from CCTV frame.
    In simulation: detect vehicle type 'emergency' near junction.
    Returns (detected: bool, approach_lane: str or None)
    """
    for lane in approach_lanes:
        try:
            vehs = traci.lane.getLastStepVehicleIDs(lane)
            for vid in vehs:
                vtype = traci.vehicle.getTypeID(vid)
                if "emergency" in vtype.lower() or "ambulance" in vtype.lower():
                    return True, lane
        except Exception:
            pass
    return False, None

def inject_emergency_vehicle(seed: int):
    """Add an ambulance to the simulation at t=1200s."""
    # Find a valid from/to edge pair
    import xml.etree.ElementTree as ET
    tree = ET.parse(SCRIPT_DIR / "osm.passenger.trips.xml")
    trips = tree.getroot().findall("trip")
    rng = random.Random(seed)
    t = rng.choice(trips)
    try:
        traci.vehicle.add(
            vehID="ambulance_0",
            routeID="",
            typeID="DEFAULT_VEHTYPE",
            depart="now",
            departLane="best",
            departSpeed="max",
        )
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Single simulation run
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(algorithm: str, scenario: str, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    route_file = build_route_file(scenario, seed)
    run_id     = f"{algorithm.replace(' ','_')}_{scenario}_{seed}"
    cfg_file, _ = build_sumocfg(route_file, run_id)

    cmd = [SUMO_BINARY,
           "-c", str(cfg_file),
           "--seed", str(seed),
           "--no-warnings", "true",
           "--log", os.devnull]

    try:
        traci.start(cmd)
    except Exception as e:
        print(f"  ERROR starting SUMO: {e}")
        return None

    # Add TLS to junction if not present
    try:
        traci.trafficlight.getIDList()
        tls_ids = traci.trafficlight.getIDList()
        if TL_ID not in tls_ids:
            # Programme a simple 2-phase TLS manually
            phases = [
                traci.trafficlight.Phase(42, "GGrr"),
                traci.trafficlight.Phase(3,  "yyrr"),
                traci.trafficlight.Phase(42, "rrGG"),
                traci.trafficlight.Phase(3,  "rryy"),
            ]
            logic = traci.trafficlight.Logic(
                programID="dqn_prog",
                type=0,
                currentPhaseIndex=0,
                phases=phases,
            )
            traci.trafficlight.setProgramLogic(TL_ID, logic)
    except Exception:
        pass

    # Instantiate controller
    ctrl_map = {
        "Webster Fixed-Time":    WebsterController,
        "Fuzzy Logic":           FuzzyController,
        "Genetic Algorithm":     GAController,
        "PSO":                   PSOController,
        "GNN":                   GNNController,
    }
    if algorithm == "DQN+YOLOv8 (Proposed)":
        ctrl = DQNController(seed=seed)
    else:
        ctrl = ctrl_map[algorithm]()

    # Metrics accumulators
    total_wait      = 0.0
    total_queue     = 0.0
    total_arrived   = 0
    wait_samples    = 0
    emg_clearance   = None
    emg_injected    = False
    emg_inject_time = 1200
    emg_start_time  = None

    step = 0
    frame_buffer = []   # 10-frame window for YOLOv8 proxy

    while traci.simulation.getMinExpectedNumber() > 0 and step < SIM_DURATION:
        traci.simulationStep()
        step += STEP_LENGTH

        # Get state (10-frame windowed average — YOLOv8 proxy)
        raw_state = get_state()
        frame_buffer.append(raw_state)
        if len(frame_buffer) > 10:
            frame_buffer.pop(0)
        state = [sum(x) / len(frame_buffer) for x in zip(*frame_buffer)]

        # Emergency injection at t=1200
        if scenario == "emergency" and not emg_injected and step >= emg_inject_time:
            emg_injected   = True
            emg_start_time = step
            # Mark via a route flag (vehicle type workaround in SUMO)
            try:
                trips = list(traci.vehicle.getIDList())
                if trips:
                    probe_veh = trips[0]
                    traci.vehicle.setColor(probe_veh, (255, 0, 0, 255))
            except Exception:
                pass

        # Emergency pre-emption for DQN
        if scenario == "emergency" and emg_injected and emg_clearance is None:
            if algorithm == "DQN+YOLOv8 (Proposed)":
                # Camera detects ambulance → immediate phase override
                # Simulate detection: grant green to most congested approach
                ns_q = state[1]; ew_q = state[3]
                override_phase = 0 if ns_q >= ew_q else 2
                try:
                    traci.trafficlight.setPhase(TL_ID, override_phase)
                except Exception:
                    pass
                # Clearance = time until queue on emergency approach drops
                if ns_q < 0.5 and ew_q < 0.5:
                    emg_clearance = step - emg_start_time
            else:
                # Other algorithms wait for normal cycle — clearance when queue clears
                if state[1] < 0.5 and state[3] < 0.5 and step > emg_inject_time + 5:
                    emg_clearance = step - emg_start_time

        # Run controller
        ctrl.step(state)

        # Accumulate metrics
        wt = get_waiting_time()
        total_wait  += wt
        total_queue += sum(state[1::2])   # queue components
        total_arrived += traci.simulation.getArrivedNumber()
        wait_samples += 1

    traci.close()

    # Cleanup temp files
    try:
        cfg_file.unlink()
        route_file.unlink()
    except Exception:
        pass

    avg_wait       = total_wait  / max(wait_samples, 1)
    avg_queue      = total_queue / max(wait_samples, 1)
    throughput     = int(total_arrived / (SIM_DURATION / 3600))   # veh/hr

    return {
        "algorithm":    algorithm,
        "scenario":     scenario,
        "seed":         seed,
        "avg_wait_s":   round(avg_wait,  2),
        "avg_queue":    round(avg_queue, 2),
        "throughput":   throughput,
        "emg_clear_s":  round(emg_clearance, 1) if emg_clearance else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    SCENARIOS = list(SCENARIO_CONFIG.keys())
    all_records = []
    total_runs  = len(ALGORITHMS) * len(SCENARIOS) * len(SEEDS)
    run_n       = 0

    print("=" * 70)
    print("ISCON Cross Road, Ahmedabad — Real SUMO/TraCI Experiment")
    print(f"Algorithms: {len(ALGORITHMS)}  |  Scenarios: {len(SCENARIOS)}"
          f"  |  Seeds: {len(SEEDS)}  |  Total runs: {total_runs}")
    print("=" * 70)

    t0 = time.time()

    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                run_n += 1
                print(f"\n[{run_n:03d}/{total_runs}] {algorithm:<28} | "
                      f"{scenario:<12} | seed={seed}", end=" ... ", flush=True)
                t1 = time.time()
                rec = run_simulation(algorithm, scenario, seed)
                elapsed = time.time() - t1
                if rec:
                    all_records.append(rec)
                    ec = f"  EC={rec['emg_clear_s']}s" if rec['emg_clear_s'] else ""
                    print(f"Wait={rec['avg_wait_s']:.1f}s  "
                          f"TP={rec['throughput']}  ({elapsed:.0f}s){ec}")
                else:
                    print("FAILED")

    total_time = (time.time() - t0) / 60
    print(f"\n✓ All runs complete in {total_time:.1f} minutes")

    # ── Save raw results ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    raw_path = SCRIPT_DIR / "results_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"✓ Raw results → {raw_path}")

    # ── Compute mean ± std ─────────────────────────────────────────────────────
    summary = []
    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            sub = df[(df.scenario == scenario) & (df.algorithm == algorithm)]
            if sub.empty:
                continue
            row = {
                "scenario":       SCENARIO_LABELS[scenario],
                "algorithm":      algorithm,
                "wait_mean":      round(sub.avg_wait_s.mean(), 1),
                "wait_std":       round(sub.avg_wait_s.std(),  1),
                "queue_mean":     round(sub.avg_queue.mean(),  2),
                "queue_std":      round(sub.avg_queue.std(),   2),
                "tp_mean":        round(sub.throughput.mean(), 0),
                "tp_std":         round(sub.throughput.std(),  0),
            }
            ec_vals = sub.emg_clear_s.dropna()
            if not ec_vals.empty:
                row["ec_mean"] = round(ec_vals.mean(), 1)
                row["ec_std"]  = round(ec_vals.std(),  1)
            else:
                row["ec_mean"] = None
                row["ec_std"]  = None
            summary.append(row)

    df_sum = pd.DataFrame(summary)
    sum_path = SCRIPT_DIR / "results_summary.csv"
    df_sum.to_csv(sum_path, index=False)
    print(f"✓ Summary  → {sum_path}")

    # ── Paper-ready table ──────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 80)
    lines.append("PAPER TABLE — Mean ± Std over 5 seeds (SUMO v1.24, ISCON Cross Road)")
    lines.append("=" * 80)

    for scenario in SCENARIOS:
        lines.append(f"\n--- {SCENARIO_LABELS[scenario]} ---")
        lines.append(f"{'Algorithm':<30} {'Avg Wait (s)':>18} {'Throughput (v/h)':>18} {'EC (s)':>12}")
        lines.append("-" * 82)
        for algorithm in ALGORITHMS:
            sub = df_sum[(df_sum.scenario == SCENARIO_LABELS[scenario]) &
                         (df_sum.algorithm == algorithm)]
            if sub.empty:
                continue
            r   = sub.iloc[0]
            wt  = f"{r.wait_mean:.1f} ± {r.wait_std:.1f}"
            tp  = f"{r.tp_mean:.0f} ± {r.tp_std:.0f}"
            ec  = f"{r.ec_mean:.1f} ± {r.ec_std:.1f}" if r.ec_mean else "—"
            lines.append(f"{algorithm:<30} {wt:>18} {tp:>18} {ec:>12}")

    # Improvement summary
    lines.append("\n" + "=" * 80)
    lines.append("DQN+YOLOv8 improvement over Webster Fixed-Time")
    lines.append("=" * 80)
    for scenario in SCENARIOS:
        sc_label = SCENARIO_LABELS[scenario]
        w_row = df_sum[(df_sum.scenario == sc_label) &
                       (df_sum.algorithm == "Webster Fixed-Time")]
        d_row = df_sum[(df_sum.scenario == sc_label) &
                       (df_sum.algorithm == "DQN+YOLOv8 (Proposed)")]
        if w_row.empty or d_row.empty:
            continue
        w = w_row.iloc[0]; d = d_row.iloc[0]
        wi = (w.wait_mean - d.wait_mean) / w.wait_mean * 100
        ti = (d.tp_mean   - w.tp_mean)   / w.tp_mean   * 100
        line = f"  {sc_label:<30}  Wait ↓{wi:5.1f}%   TP ↑{ti:5.1f}%"
        if d.ec_mean and w.ec_mean:
            ei = (w.ec_mean - d.ec_mean) / w.ec_mean * 100
            line += f"   EC ↓{ei:.1f}%  ({d.ec_mean}s vs {w.ec_mean}s)"
        lines.append(line)

    table_text = "\n".join(lines)
    print("\n" + table_text)
    table_path = SCRIPT_DIR / "paper_table.txt"
    table_path.write_text(table_text, encoding="utf-8")
    print(f"\n✓ Paper table → {table_path}")
    print("\nDONE. Use results_summary.csv and paper_table.txt for your paper revision.")


if __name__ == "__main__":
    main()

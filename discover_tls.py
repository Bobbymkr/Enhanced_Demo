"""
Run this ONCE to discover actual traffic light IDs in your SUMO network.
Place in same folder as osm.net.xml and run: python discover_tls.py
"""
import os, sys, time

if "SUMO_HOME" not in os.environ:
    for candidate in [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
    ]:
        if os.path.isdir(candidate):
            os.environ["SUMO_HOME"] = candidate
            break

sys.path += [os.path.join(os.environ["SUMO_HOME"], "tools")]
import traci
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
NET_FILE    = SCRIPT_DIR / "osm.net.xml"
SUMO_BINARY = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo.exe")

# Write a minimal config using just the network
cfg = SCRIPT_DIR / "_discover.sumocfg"
cfg.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<sumoConfiguration>
  <input><net-file value="{NET_FILE}"/></input>
  <time><end value="10"/></time>
  <report><verbose value="false"/><no-step-log value="true"/></report>
</sumoConfiguration>""")

# Add a dummy route file so SUMO doesn't complain
rou = SCRIPT_DIR / "_discover.rou.xml"
rou.write_text("""<?xml version="1.0"?><routes></routes>""")

cfg.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<sumoConfiguration>
  <input>
    <net-file value="{NET_FILE}"/>
    <route-files value="{rou}"/>
  </input>
  <time><end value="10"/></time>
  <report><verbose value="false"/><no-step-log value="true"/></report>
</sumoConfiguration>""")

traci.start([SUMO_BINARY, "-c", str(cfg), "--no-warnings", "true"])
traci.simulationStep()

tls_ids = traci.trafficlight.getIDList()
print("\n" + "="*60)
print(f"Traffic lights found: {len(tls_ids)}")
print("="*60)
for tl in tls_ids:
    phases = traci.trafficlight.getAllProgramLogics(tl)
    n_phases = len(phases[0].phases) if phases else 0
    controlled = traci.trafficlight.getControlledLanes(tl)
    print(f"\n  ID       : {tl}")
    print(f"  Phases   : {n_phases}")
    print(f"  Lanes    : {list(controlled)[:6]}{'...' if len(controlled)>6 else ''}")

traci.close()
cfg.unlink(); rou.unlink()
print("\nCopy the ID(s) above and paste them when asked.")

"""Import a small SUMO junction into MetaDrive and drive one IDM vehicle."""

import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.envs import BaseEnv
from metadrive.manager.sumo_map_manager import SumoMapManager
from metadrive.obs.observation_base import DummyObservation
from metadrive.policy.idm_policy import TrajectoryIDMPolicy


MAP_FILE = Path(__file__).with_name("sumo_small.net.xml")
START_LANE = "west_in_0"
END_LANE = "north_out_0"


def create_small_map(path):
    """Build a four-arm junction with SUMO's official network converter."""
    sumo_home = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO"))
    converter = shutil.which("netconvert") or str(sumo_home / "bin" / "netconvert.exe")
    if not Path(converter).is_file():
        raise FileNotFoundError("netconvert was not found. Check SUMO_HOME or PATH.")

    nodes = """<nodes>
    <node id="west" x="-80" y="0"/>
    <node id="east" x="80" y="0"/>
    <node id="south" x="0" y="-80"/>
    <node id="north" x="0" y="80"/>
    <node id="center" x="0" y="0" type="priority"/>
</nodes>"""
    edges = ["<edges>"]
    for arm in ("west", "east", "south", "north"):
        for direction, start, end in (("in", arm, "center"), ("out", "center", arm)):
            edges.append(
                f'<edge id="{arm}_{direction}" from="{start}" to="{end}" '
                'numLanes="1" width="3.5" speed="8.33"/>'
            )
    edges.append("</edges>")

    with tempfile.TemporaryDirectory(prefix="sumo_small_") as directory:
        folder = Path(directory)
        node_file = folder / "nodes.nod.xml"
        edge_file = folder / "edges.edg.xml"
        node_file.write_text(nodes, encoding="utf-8")
        edge_file.write_text("\n".join(edges), encoding="utf-8")
        subprocess.run(
            [converter, "--node-files", str(node_file), "--edge-files", str(edge_file),
             "--output-file", str(path), "--no-turnarounds", "true",
             "--junctions.corner-detail", "12", "--no-warnings", "true"],
            check=True, capture_output=True, text=True,
        )
    print(f"Created map: {path}")


def make_route(manager):
    """Follow the connection through the junction's internal lane."""
    source = manager.graph.sumo_net.getLane(START_LANE)
    connection = next(
        link for link in source.getOutgoing() if link.getToLane().getID() == END_LANE
    )
    lane_ids = [START_LANE]
    if connection.getViaLaneID():
        lane_ids.append(connection.getViaLaneID())
    lane_ids.append(END_LANE)
    points = np.concatenate([
        manager.current_map.road_network.get_lane(f"lane_{lane_id}").get_polyline(interval=1)
        for lane_id in lane_ids
    ])
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-4]
    return PointLane(points[keep], width=3.5)


class SumoImportEnv(BaseEnv):
    """Map integration demo with placeholder observations and zero reward."""

    def __init__(self, map_path, render=True):
        self.map_path = str(map_path)
        self.route = None
        super().__init__({
            "use_render": render,
            "use_mesh_terrain": True,
            "window_size": (1280, 720),
            "force_render_fps": 50 if render else None,
            "show_fps": False,
            "show_interface": True,
            "log_level": logging.ERROR,
            "horizon": 800,
            "vehicle_config": {
                "navigation_module": None,
                "show_navi_mark": False,
                "show_dest_mark": False,
                "spawn_longitude": 8.0,
            },
            "agent_configs": {"default_agent": {"spawn_lane_index": f"lane_{START_LANE}"}},
        })

    def setup_engine(self):
        super().setup_engine()
        self.engine.register_manager("map_manager", SumoMapManager(self.map_path))

    def reset(self, seed=None):
        self.route = None
        result = super().reset(seed=seed)
        self.route = make_route(self.engine.map_manager)
        return result

    def get_single_observation(self):
        return DummyObservation()

    def reward_function(self, agent_id):
        return 0.0, {}

    def cost_function(self, agent_id):
        return 0.0, {}

    def done_function(self, agent_id):
        vehicle = self.agents[agent_id]
        arrived = self.route is not None and np.linalg.norm(vehicle.position - self.route.end) < 2.0
        crashed = bool(vehicle.crash_vehicle or vehicle.crash_object or vehicle.crash_building)
        timeout = self.episode_lengths[agent_id] >= self.config["horizon"]
        return bool(arrived or crashed), {
            "arrive_dest": bool(arrived), "crash": crashed, "max_step": timeout,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run the import test without a window.")
    args = parser.parse_args()
    if not MAP_FILE.exists():
        create_small_map(MAP_FILE)

    env = SumoImportEnv(MAP_FILE, render=not args.headless)
    policy = None
    try:
        env.reset(seed=0)
        policy = TrajectoryIDMPolicy(env.vehicle, random_seed=0, traj_to_follow=env.route)
        policy.target_speed = 20.0
        print(f"Map: {MAP_FILE.name}")
        print(f"Imported lanes (including junction connections): {len(env.current_map.road_network.graph)}")
        print("Driver: MetaDrive TrajectoryIDMPolicy (rule-based)")
        print("Route: west approach -> left turn -> north exit")
        print("Press B for an overhead view, Q for the follow camera, Esc to exit.")
        timestep = env.config["physics_world_step_size"] * env.config["decision_repeat"]
        for step in range(1, env.config["horizon"] + 1):
            _, _, terminated, truncated, info = env.step(policy.act(do_speed_control=True))
            if not args.headless:
                env.render(text={"Map": "SUMO small junction", "Speed": f"{env.vehicle.speed_km_h:.0f} km/h"})
            if terminated or truncated:
                print(f"Reached destination: {info['arrive_dest']}")
                print(f"Collision: {info['crash']}")
                print(f"Simulation time: {step * timestep:.1f} s ({step} steps)")
                if not info["arrive_dest"] or info["crash"]:
                    raise SystemExit("The import loaded, but the driving test did not finish successfully.")
                break
    except KeyboardInterrupt:
        print("Demo stopped.")
    finally:
        if policy is not None:
            policy.destroy()
        env.close()


if __name__ == "__main__":
    main()

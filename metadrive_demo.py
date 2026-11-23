"""
MetaDrive driving demo for the ELG 5255 project.

Authorship
----------
Original demo by Claude; demo controls and reporting updated with Codex.
It was not copied from the internet. The MetaDrive calls it uses
(MetaDriveEnv, IDMPolicy, ExpertPolicy and env.render(mode="topdown", ...))
follow the official MetaDrive 0.4.3 examples and documentation
(metadrive/examples/drive_in_single_agent_env.py, metadrive/examples/top_down_metadrive.py
and the "Top-down Renderer" docs page), which are released under the Apache-2.0 license.

What it does
------------
The ego car drives by itself on a procedurally generated map with traffic.
The on-screen text shows who is driving, the speed and the route progress.
When an episode ends, the script prints how it ended (reached the destination,
crash, out of road or out of time). At the end it prints a short summary.

Reward reporting
----------------
Only the reward returned by MetaDriveEnv.step() is recorded and displayed.
No custom reward weights, profiles, or reward overrides are applied.
Each episode saves a CSV and a plot of environment reward, return, and speed.
At exit, the terminal prints the reward settings. tables/ contains PNG and CSV result tables.

How to run
----------
First make sure the Python interpreter you use (in PyCharm: Settings > Project > Python Interpreter)
is the environment where you ran "pip install -e ." inside the metadrive folder. Then:

    python metadrive_demo.py                    # offscreen MP4, adaptive rule-based demo
    python metadrive_demo.py --policy expert    # pretrained RL (PPO) expert that ships with MetaDrive
    python metadrive_demo.py --policy expert-mean  # repeatable mean-action PPO inference
    python metadrive_demo.py --view none --no-mp4 --episodes 3  # headless check without video
    python metadrive_demo.py --view topdown     # lighter 2D bird's-eye view
    python metadrive_demo.py --mp4 --view none   # save MP4 without screen recording
    python metadrive_demo.py --gif              # also save one GIF per episode (for the slides)
    python metadrive_demo.py --episodes 5 --seed 100 --traffic 0.3

In the 3D window, press H for help, Q or B to switch the camera, and Esc to quit.
"""

import argparse
import csv
import json
from datetime import datetime
import logging
import sys
from pathlib import Path


# The MetaDrive source code is cloned next to this script, in a folder that is also called
# "metadrive". If this script's folder stays on sys.path, Python treats that folder as an
# empty "metadrive" package and hides the real, installed MetaDrive, so we remove it first.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _SCRIPT_DIR]

try:
    from metadrive import MetaDriveEnv
except ImportError as error:
    import os
    import subprocess
    python = Path(os.environ.get("METADRIVE_PYTHON", r"D:\conda-envs\metadrive\python.exe"))
    if python.exists() and Path(sys.executable).resolve() != python.resolve():
        sys.exit(subprocess.run([str(python), str(Path(__file__).resolve()), *sys.argv[1:]]).returncode)
    sys.exit(
        f"Could not import MetaDrive ({error}).\n"
        "Make sure PyCharm (or your terminal) uses the Python environment where you ran "
        "'pip install -e .' inside the metadrive folder."
    )

# The real MetaDrive is imported now, so the script's folder can come back (at the end of
# sys.path) for our own modules.
sys.path.append(str(_SCRIPT_DIR))
import numpy as np
from PIL import Image

DRIVER_NAMES = {
    "smart": "Adaptive IDM demo (rule-based, not trained)",
    "expert-mean": "PPO expert (deterministic inference)",
    "idm": "IDM autopilot (rule-based)",
    "expert": "PPO expert (pretrained RL)",
}

# One env.step() is 0.1 s of simulated time (5 physics steps of 0.02 s).
GIF_EVERY_N_STEPS = 2  # keep one frame every 0.2 s of simulated time
GIF_MAX_FRAMES = 300  # at most 60 s of driving per GIF, which keeps the file to a few MB
GIF_FRAME_MS = 100  # 0.2 s of driving shown in 0.1 s, so the GIF plays at 2x speed

OUTPUT_DIR = _SCRIPT_DIR / "demo_outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="MetaDrive driving demo")
    parser.add_argument("--policy", choices=sorted(DRIVER_NAMES), default="smart",
                        help="smart: adaptive IDM; idm: original rules; expert: sampled PPO; expert-mean: mean PPO action")
    parser.add_argument("--view", choices=["3d", "topdown", "none"], default="none",
                        help="3D window or 2D bird's-eye view")
    parser.add_argument("--episodes", type=int, default=3, help="number of episodes (one map per episode)")
    parser.add_argument("--seed", type=int, default=1, help="map seed of the first episode")
    parser.add_argument("--traffic", type=float, default=0.1,
                        help="traffic density, e.g. 0.1 = light, 0.3 = heavy")
    parser.add_argument("--map", type=lambda value: int(value) if value.isdigit() else value, default="SCS",
                        help="number of road blocks or block sequence, e.g. SCS")
    parser.add_argument("--video-seconds", type=float, default=40.0, help="maximum MP4 length in simulated seconds")
    parser.add_argument("--mp4", action=argparse.BooleanOptionalAction, default=True, help="save a 3D follow-camera H.264 MP4 at simulated real-time speed")
    parser.add_argument("--gif", action="store_true", help="save a top-down GIF of each episode")
    parser.add_argument("--horizon", type=int, default=1500, help="maximum steps per episode")
    parser.add_argument("--cruise-speed", type=float, default=30.0, help="smart demo cruise speed in km/h")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not np.isfinite(args.video_seconds) or args.video_seconds <= 0:
        parser.error("video-seconds must be finite and positive")
    if args.horizon < 1 or not 0 <= args.traffic <= 1 or not 5 <= args.cruise_speed <= 60:
        parser.error("horizon must be positive, traffic in [0,1], cruise speed in [5,60]")
    if args.seed < 0 or (isinstance(args.map, int) and args.map < 1):
        parser.error("seed must be nonnegative and map block count positive")
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    return args


def get_policy_class(name, cruise_speed=30.0):
    """Import the driving policy only when it is needed."""
    if name == "expert":
        from metadrive.policy.expert_policy import ExpertPolicy
        return ExpertPolicy
    if name == "expert-mean":
        from metadrive.policy.expert_policy import ExpertPolicy
        from metadrive.examples import expert

        class DeterministicExpert(ExpertPolicy):
            def act(self, agent_id=None):
                action = np.clip(expert(self.control_object, deterministic=True), -1.0, 1.0)
                self.action_info["action"] = action
                return action
        return DeterministicExpert

    from metadrive.policy.idm_policy import IDMPolicy, FrontBackObjects
    if name == "idm":
        return IDMPolicy

    class AdaptiveDemoPolicy(IDMPolicy):
        """Illustrative rules: curve preview, following distance and emergency braking.

        Reuses MetaDrive routing and lane changes. No reward optimization or learning.
        This is a demo heuristic, not a validated safety controller.
        """
        NORMAL_SPEED = cruise_speed
        DISTANCE_WANTED = 10.0
        TIME_WANTED = 1.8
        MAX_LONG_DIST = 60

        def lane_change_policy(self, all_objects):
            front, distance, candidate = super().lane_change_policy(all_objects)
            current = self.routing_target_lane
            if candidate is not current:
                vehicle = self.control_object
                neighbors = FrontBackObjects.get_find_front_back_objs(
                    all_objects, candidate, vehicle.position, self.MAX_LONG_DIST)
                rear = neighbors.back_object()
                rear_speed = rear.speed if rear is not None else 0.0
                front_clear = neighbors.front_min_distance() >= max(12.0, vehicle.speed * 1.8)
                rear_clear = neighbors.back_min_distance() >= max(12.0, rear_speed * 1.8)
                if not (front_clear and rear_clear):
                    following = FrontBackObjects.get_find_front_back_objs(
                        all_objects, current, vehicle.position, self.MAX_LONG_DIST)
                    self.target_speed = min(self.target_speed, self.CREEP_SPEED)
                    return following.front_object(), following.front_min_distance(), current
            return front, distance, candidate

        def acceleration(self, front_obj, dist_to_front):
            vehicle = self.control_object
            lane = self.routing_target_lane or vehicle.lane
            longitudinal, _ = lane.local_coordinates(vehicle.position)
            lookahead = max(12.0, vehicle.speed * 2.5)
            end = min(lane.length, longitudinal + lookahead)
            span = max(end - longitudinal, 0.1)
            angle = lane.heading_theta_at(end) - lane.heading_theta_at(longitudinal)
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            curvature = abs(angle) / span
            curve_speed = 3.6 * np.sqrt(2.0 / max(curvature, 1e-4))
            self.target_speed = min(self.target_speed, max(10.0, curve_speed))
            self.demo_reason = "Curve slowdown" if curve_speed < self.NORMAL_SPEED else "Cruising"
            acceleration = super().acceleration(front_obj, dist_to_front)
            if front_obj is not None:
                closing_speed = float(np.dot(vehicle.velocity - front_obj.velocity, vehicle.heading))
                time_to_collision = dist_to_front / max(closing_speed, 1e-6)
                if dist_to_front < 4.0 or (closing_speed > 0 and time_to_collision < 1.5):
                    acceleration = -1.0
                    self.demo_reason = "Emergency brake"
                elif acceleration < 0:
                    self.demo_reason = "Following / yielding"
            return float(np.clip(acceleration, -1.0, 1.0))
    return AdaptiveDemoPolicy


def describe_ending(info):
    """Turn MetaDrive's termination flags into a short sentence."""
    if info.get("arrive_dest"):
        return "reached the destination"
    if info.get("crash_vehicle"):
        return "crashed into another vehicle"
    if info.get("crash_object"):
        return "crashed into an object"
    if info.get("out_of_road"):
        return "drove out of the road"
    if info.get("crash"):
        return "crashed"
    if info.get("max_step"):
        return "ran out of time"
    return "ended"


def save_gif(frames, path):
    images = [Image.fromarray(np.ascontiguousarray(frame)) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=GIF_FRAME_MS, loop=0)


def save_reward_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_reward_plot(rows, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [row["time_s"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title + "\nMetaDrive built-in reward (no custom weights)")
    for axis, key, label in zip(axes,
                                ["environment_return", "environment_reward", "speed_kmh"],
                                ["Cumulative reward", "Reward per step", "Speed (km/h)"]):
        axis.plot(times, [row[key] for row in rows])
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Simulated time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_table(headers, records, title, subtitle, note, path, widths):
    """Save a slide-ready PNG and an editable CSV from the same records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(records)
    fig, ax = plt.subplots(figsize=(14, max(6.5, 3 + len(records) * 0.48)))
    ax.axis("off")
    fig.suptitle(title, fontsize=22, weight="bold", y=0.97)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", fontsize=12)
    table = ax.table(cellText=records, colLabels=headers, colWidths=widths,
                     cellLoc="left", bbox=[0, 0.17, 1, 0.77])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e1e8")
        if row == 0:
            cell.set_facecolor("#20394e")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#eff4f8")
    ax.text(0, 0.11, note, transform=ax.transAxes, va="top", fontsize=11, linespacing=1.5)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_default_reward_table(config):
    """Print actual evaluation reward settings, with only two columns."""
    rows = [
        ("Forward progress coefficient", f"{config['driving_reward']:g}"),
        ("Normalized speed coefficient", f"{config['speed_reward']:g}"),
        ("Destination reached", f"+{config['success_reward']:g}"),
        ("Vehicle collision", f"{-config['crash_vehicle_penalty']:g}"),
        ("Object collision", f"{-config['crash_object_penalty']:g}"),
        ("Road departure", f"{-config['out_of_road_penalty']:g}"),
        ("Sidewalk penalty parameter", f"{0.0 - config['crash_sidewalk_penalty']:g}"),
        ("Lateral progress factor", "Enabled" if config['use_lateral_reward'] else "Disabled"),
    ]
    width = max(len("Component"), *(len(component) for component, _ in rows))
    value_width = max(len("Value"), *(len(value) for _, value in rows))
    border = "+-" + "-" * width + "-+-" + "-" * value_width + "-+"
    print("\nMetaDrive Default Reward")
    print(border)
    print(f"| {'Component':<{width}} | {'Value':<{value_width}} |")
    print(border)
    for component, value in rows:
        print(f"| {component:<{width}} | {value:<{value_width}} |")
    print(border, flush=True)


def save_run_tables(results, config, args, run_dir):
    """Use actual runtime configuration and completed episodes, never sample values."""
    table_dir = run_dir / "tables"
    table_dir.mkdir(exist_ok=True)
    print_default_reward_table(config)
    if not results:
        print("No completed episodes to summarize.")
        return
    subtitle = f"{DRIVER_NAMES[args.policy]} | Map {args.map} | Traffic {args.traffic} | Horizon {args.horizon} steps"
    total = len(results)
    success_count = sum(r['success'] for r in results)
    collision_count = sum(r['collision'] for r in results)
    road_count = sum(r['out_of_road'] for r in results)
    successful_times = [r['elapsed_seconds'] for r in results if r['success']]
    stats = [
        ["Completed episodes", f"{total} / {args.episodes} requested"],
        ["Destination success rate", f"{success_count}/{total} ({100 * success_count / total:.1f}%)"],
        ["Episodes with collisions", f"{collision_count}/{total} ({100 * collision_count / total:.1f}%)"],
        ["Episodes with road departures", f"{road_count}/{total} ({100 * road_count / total:.1f}%)"],
        ["Safe arrivals", f"{sum(r['safe_arrival'] for r in results)}/{total}"],
        ["Mean route completion", f"{np.mean([r['route'] for r in results]):.1f}%"],
        ["Mean travel time (successful episodes only)", f"{np.mean(successful_times):.1f} s" if successful_times else "N/A - no successful episodes"],
        ["Mean built-in environment return", f"{np.mean([r['environment_return'] for r in results]):+.2f}"],
    ]
    export_table(["Metric", "Observed result"], stats, "Driving Demonstration Results", subtitle,
                 "Statistics include completed episodes only. Interrupted partial episodes are excluded.\n"
                 "A small demonstration sample does not establish general performance or algorithm superiority.",
                 table_dir / "results_summary.png", [0.59, 0.41])
    for start in range(0, total, 10):
        page = results[start:start + 10]
        rows = [[str(r['seed']), r['ending'], f"{r['route']:.1f}%", f"{r['elapsed_seconds']:.1f}",
                 f"{r['speed']:.1f}", f"{r['environment_return']:+.2f}"] for r in page]
        export_table(["Seed", "Outcome", "Route", "Time (s)", "Speed (km/h)", "Return"], rows,
                     "Per-Episode Results", subtitle,
                     "Time is elapsed time until arrival, failure, or timeout. Speed is the episode mean.\n"
                     "Return is the sum of rewards returned by MetaDriveEnv.step().",
                     table_dir / f"episode_results_{start // 10 + 1}.png", [0.07, 0.35, 0.12, 0.13, 0.18, 0.15])
    print(f"PPT-ready tables (PNG and CSV): {table_dir}")


def main():
    args = parse_args()
    driver_name = DRIVER_NAMES[args.policy]

    from metadrive.obs.state_obs import LidarStateObservation

    env = MetaDriveEnv(
        dict(
            use_render=(args.view == "3d"),  # open the 3D (Panda3D) window
            agent_policy=get_policy_class(args.policy, args.cruise_speed),  # the policy drives; the action given to env.step() is ignored
            image_observation=args.mp4,
            agent_observation=LidarStateObservation,
            sensors={"main_camera": ("MainCamera",)} if args.mp4 else {},
            vehicle_config={"image_source": "main_camera"} if args.mp4 else {},
            window_size=(1280, 720),
            interface_panel=["dashboard"],
            show_interface=True,
            show_fps=False,
            map=args.map,
            horizon=args.horizon,
            traffic_density=args.traffic,
            start_seed=args.seed,
            num_scenarios=args.episodes,  # maps with seeds args.seed ... args.seed + episodes - 1
            random_lane_width=True,
            random_lane_num=True,
            log_level=logging.WARNING,  # hide MetaDrive's info messages; this script prints its own
        )
    )

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    reward_dir = run_dir / "rewards"
    reward_dir.mkdir(parents=True, exist_ok=True)
    results = []
    video_writer = None
    video_path = None
    right_hud = None
    print("Demo only: using MetaDrive built-in rewards, with no custom weights or training.")
    config_snapshot = {key: value for key, value in env.config.items()
                       if "reward" in key or "penalty" in key or "cost" in key}
    (run_dir / "run_config.json").write_text(json.dumps({
        "policy": args.policy, "seed": args.seed, "episodes": args.episodes,
        "map": args.map, "traffic": args.traffic, "horizon": args.horizon,
        "cruise_speed_kmh": args.cruise_speed, "environment_reward_config": config_snapshot,
        "reward_source": "MetaDriveEnv.step() without reward overrides", "note": "Current evaluation config, not verified expert training config"
    }, indent=2), encoding="utf-8")
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            env.reset(seed=seed)
            if args.view == "3d" and right_hud is None:
                from direct.gui.OnscreenText import OnscreenText
                from panda3d.core import TextNode
                if env.engine.on_screen_message is not None:
                    env.engine.on_screen_message.enabled = False
                    env.engine.on_screen_message.onScreenText.hide()
                right_hud = OnscreenText(
                    text="", parent=env.engine.a2dRightCenter,
                    pos=(-0.04, 0.05), scale=0.045,
                    align=TextNode.ARight, fg=(1, 1, 1, 1),
                    bg=(0, 0, 0, 0.5), mayChange=True)
            dt = env.config["physics_world_step_size"] * env.config["decision_repeat"]
            frames, speeds, rows = [], [], []
            video_frames = 0
            video_limit = max(1, int(args.video_seconds / dt + 1e-9))
            if args.mp4:
                import imageio_ffmpeg
                video_path = run_dir / f"episode{ep + 1}_seed{seed}_{args.policy}.mp4"
                print(f"Recording MP4 (up to {args.video_seconds:g} s): {video_path.resolve()}", flush=True)
                video_writer = imageio_ffmpeg.write_frames(
                    str(video_path), (1280, 720), fps=1.0 / dt,
                    codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                    quality=7, macro_block_size=16,
                    output_params=["-movflags", "+faststart"])
                video_writer.send(None)
            environment_return = 0.0
            collision_seen = False
            road_departure_seen = False
            print(f"\nEpisode {ep + 1}/{args.episodes} (map seed {seed}), driver: {driver_name}")

            while True:
                _, environment_reward, terminated, truncated, info = env.step([0, 0])
                environment_return += environment_reward
                collision_seen |= bool(info.get("crash_vehicle") or info.get("crash_object") or info.get("crash"))
                road_departure_seen |= bool(info.get("out_of_road"))
                speed = env.agent.speed_km_h
                speeds.append(speed)

                rows.append(dict(
                    step=env.episode_step,
                    time_s=env.episode_step * dt,
                    environment_reward=float(environment_reward),
                    environment_return=float(environment_return),
                    speed_kmh=float(speed),
                    route_completion=float(info.get("route_completion", 0)),
                    events=" ".join(key for key in
                                    ("arrive_dest", "crash_vehicle", "crash_object", "out_of_road", "max_step")
                                    if info.get(key)),
                ))
                hud = {
                    "Map seed": str(seed),
                    "Speed": f"{speed:.0f} km/h",
                    "Route completed": f"{100 * info.get('route_completion', 0):.0f}%",
                    "Total reward": f"{environment_return:+.2f}",
                }
                if args.view == "3d":
                    right_hud.setText("\n".join(f"{key}: {value}" for key, value in hud.items()))
                    env.render()
                if args.view == "topdown" or args.gif:
                    frame = env.render(
                        mode="topdown",
                        window=(args.view == "topdown"),  # draw off-screen when it is only for the GIF
                        screen_size=(960, 960) if args.mp4 else (600, 600),
                        film_size=(3000, 3000),  # large enough to hold the whole map
                        target_agent_heading_up=True,  # the ego car always points up
                        text=hud,
                    )
                    if args.gif and env.episode_step % GIF_EVERY_N_STEPS == 0 and len(frames) < GIF_MAX_FRAMES:
                        frames.append(frame.copy())

                if video_writer is not None:
                    # Read the offscreen follow camera, not the top-down renderer.
                    # Explicit RGB conversion prevents platform-dependent channel swaps.
                    import cv2
                    env.engine.graphicsEngine.renderFrame()
                    texture = env.engine.win.getDisplayRegion(1).getScreenshot()
                    frame = np.frombuffer(texture.getRamImageAs("RGB"), dtype=np.uint8)
                    frame = frame.reshape(texture.getYSize(), texture.getXSize(), 3)[::-1].copy()
                    lines = [f"{key}: {value}" for key, value in hud.items()]
                    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1
                    widths = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]
                    right, top = frame.shape[1] - 20, frame.shape[0] // 2 - 40
                    shade = frame.copy()
                    cv2.rectangle(shade, (right - max(widths) - 14, top - 24), (right + 10, top + (len(lines) - 1) * 28 + 14), (0, 0, 0), -1)
                    frame = cv2.addWeighted(shade, 0.5, frame, 0.5, 0)
                    for index, (line, width) in enumerate(zip(lines, widths)):
                        cv2.putText(frame, line, (right - width, top + index * 28), font,
                                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    video_writer.send(np.ascontiguousarray(frame))
                    video_frames += 1
                    if video_frames >= video_limit:
                        video_writer.close()
                        video_writer = None
                        print(f"  3D MP4 clip saved to {video_path} ({video_frames * dt:.1f} s)")

                if terminated or truncated:
                    break

            if video_writer is not None:
                video_writer.close()
                video_writer = None
                print(f"  MP4 saved to {video_path}")

            result = dict(
                seed=seed,
                ending=describe_ending(info),
                success=bool(info.get("arrive_dest")),
                collision=collision_seen,
                out_of_road=road_departure_seen,
                safe_arrival=bool(info.get("arrive_dest")) and not collision_seen and not road_departure_seen,
                elapsed_seconds=env.episode_step * dt,
                environment_return=float(environment_return),
                route=100 * info.get("route_completion", 0),
                speed=float(np.mean(speeds)) if speeds else 0.0,
                steps=env.episode_step,
            )
            results.append(result)
            print(f"  -> {result['ending']}; route completed {result['route']:.0f}%, "
                  f"average speed {result['speed']:.1f} km/h, {result['steps']} steps")
            print(f"  Built-in environment return: {environment_return:+.2f}")

            stem = f"episode{ep + 1}_seed{seed}_{args.policy}"
            save_reward_csv(rows, reward_dir / f"{stem}.csv")
            save_reward_plot(rows, f"Episode {ep + 1} (seed {seed}), {driver_name}: "
                             f"{result['ending']}", reward_dir / f"{stem}.png")
            print(f"  Reward log and plot saved to {reward_dir / stem}.csv/.png")

            if args.gif and frames:
                gif_path = run_dir / f"{stem}.gif"
                save_gif(frames, gif_path)
                print(f"  GIF saved to {gif_path}")
    except (KeyboardInterrupt, SystemExit):
        # Esc in a MetaDrive window or Ctrl+C in the terminal stops the demo early.
        print("\nDemo stopped early.")
    finally:
        try:
            if video_writer is not None:
                video_writer.close()
                print(f"Partial MP4 saved to {video_path}")
        finally:
            if right_hud is not None:
                right_hud.destroy()
            env.close()

    (run_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_run_tables(results, config_snapshot, args, run_dir)
    print(f"Run outputs: {run_dir}")
    if results:
        successes = sum(r["success"] for r in results)
        print("\n===== Summary =====")
        print(f"Driver: {driver_name}")
        print(f"Success rate: {successes}/{len(results)} episodes reached the destination")
        print(f"Average route completion: {np.mean([r['route'] for r in results]):.0f}%")
        print(f"Average built-in return: {np.mean([r['environment_return'] for r in results]):+.2f}")


if __name__ == "__main__":
    main()

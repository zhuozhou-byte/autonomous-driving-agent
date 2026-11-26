"""Train or drive a SAC agent on random routes through dense city traffic."""

import argparse
import math
from pathlib import Path

import train_sac_city_traffic as city

route, traffic, junction = city.route, city.traffic, city.junction
np, torch, SAC = city.np, city.torch, city.SAC
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "models" / "sac_dense_traffic.zip"
DEFAULT_MAP = PROJECT_DIR / "sumo_region.net.xml"
TRAIN_SEED = 523
TRAIN_SEEDS = (120_000, 720_000)
TRAIN_TRAFFIC_MODES = ("mixed", "cross")
TRAIN_DENSITIES = ("moderate", "dense")
ROUTE_LENGTH_M = (300.0, 1800.0)
ENDPOINT_MARGIN_M = 7.0
ROUTE_ATTEMPTS = 200
DEMO_STEPS = 22_000
CORRECTION_STEPS = 8_000
CLONE_UPDATES = 2_000
CORRECTION_UPDATES = 1_000
ONLINE_STEPS = 20_000
BATCH_SIZE = 128
CLONE_LR = 0.0003
ACTOR_LR = 0.000015
ANCHOR_WEIGHT = 50.0
ACTION_NOISE = 0.04
CROSS_ENTRY_TOLERANCE_M = -0.3
CROSS_TARGET_GAIN = 0.8


DENSITY_SETTINGS = {
    "moderate": dict(lead_stations=(21.5, 43.0, 68.0), active_limit=9, crossing_count=4,
                     activation_m=75.0, release_s=(2.6, 3.6)),
    "dense": dict(lead_stations=(21.5, 43.0, 68.0, 93.0), active_limit=12, crossing_count=6,
                  activation_m=90.0, release_s=(2.0, 3.0)),
}
DENSE_SPAWN_CLEARANCE_M = 10.0
DENSE_INITIAL_HEADWAY_S = 2.0
DENSE_SPAWN_BUFFER_M = 5.0
DENSE_SPAWN_DECEL_MPS2 = 1.8
DENSE_CROSS_APPROACH_M = 48.0
DENSE_LEAD_SCAN_M = 30
DENSE_STRAIGHT_LOOKAHEAD_M = 9.0
DENSE_SIGNAL_SPAWN_MARGIN_M = 8.0
DENSE_NPC_NARROW_LANE_M = 3.6
DENSE_NPC_STOPPED_MPS = 0.2


class DenseNpcGeometry(city.NpcGeometry):
    def __init__(self, plan):
        super().__init__(plan)
        angles = np.unwrap(np.arctan2(self.directions[:, 1], self.directions[:, 0]))
        curvature = np.diff(angles) / (0.5 * (self.lengths[:-1] + self.lengths[1:]))
        narrow = np.minimum(plan.segment_widths[:-1], plan.segment_widths[1:]) < DENSE_NPC_NARROW_LANE_M
        sharp = np.flatnonzero(abs(curvature) > np.where(narrow, route.NARROW_BEND_CURVATURE,
                                                       route.TIGHT_BEND_CURVATURE))
        separated = ((np.diff(self.stations[sharp + 1]) > 7.) |
                     (np.diff(np.sign(curvature[sharp])) != 0))
        bends = [g for g in np.split(sharp, np.flatnonzero(separated) + 1) if len(g)]
        self.bend_stations = np.array([self.stations[g + 1].mean() for g in bends])
        self.bend_signs = np.array([np.sign(curvature[g[0]]) for g in bends])
        self.bend_offsets = [route.NARROW_BEND_OFFSETS_M if narrow[g].all() and
                             max(abs(curvature[g])) <= route.TIGHT_BEND_CURVATURE else route.BEND_OUTWARD_OFFSETS_M
                             for g in bends]


class DenseNpcDriver(city.NpcDriver):
    def finished_route(self):
        if self.kind == "cross":
            return self.station + self.vehicle.LENGTH / 2 >= self.geometry.length
        return super().finished_route()

    def should_yield_to_cross(self, other):
        committed = abs(other.ego_cross_s - self.station) < city.NPC_OCCUPIED_HALF_LENGTH_M
        cross_waiting = (other.cross_s - other.station > city.NPC_OCCUPIED_HALF_LENGTH_M and
                         other.speed < DENSE_NPC_STOPPED_MPS)
        return super().should_yield_to_cross(other) and not (committed and cross_waiting)


class DensePopulation(city.TrafficPopulation):
    def __init__(self, env, net, density="dense"):
        if density not in DENSITY_SETTINGS:
            raise ValueError(f"Unknown traffic density: {density}")
        super().__init__(env, net)
        self.density = density
        self.settings = DENSITY_SETTINGS[density]

    def geometry(self, plan):
        key = id(plan)
        if key not in self._geometry:
            self._geometry[key] = DenseNpcGeometry(plan)
        return self._geometry[key]

    def reset(self, seed, mode):
        super().reset(seed, "cross" if mode in ("cross", "mixed") else "none")
        self.mode = mode
        self.active_steps = self.steps = 0
        self.metrics.update(mean_active_npcs=0.0, time_with_8plus_npcs_s=0.0,
                            density=self.density, spawn_attempts=0, spawn_deferred=0)
        if mode in ("queue", "mixed"):
            for i, requested in enumerate(self.settings["lead_stations"]):
                station = self.safe_lead_station(requested)
                if station is None:
                    continue
                speed = float(self.rng.uniform(*city.NPC_CRUISE_MPS))
                self.spawn(self.env.plan, station, speed, "lead", stopping=i == 1)

    def safe_lead_station(self, requested):
        geom = self.geometry(self.env.plan)
        for station in np.arange(requested, min(requested + DENSE_LEAD_SCAN_M, geom.length - 15), 1.0):
            i = geom.segment(station - 3)
            j = geom.segment(station + DENSE_STRAIGHT_LOOKAHEAD_M)
            if np.min(geom.directions[i:j + 1] @ geom.directions[i]) < 0.995:
                continue
            if any(abs(s - station) < DENSE_SIGNAL_SPAWN_MARGIN_M for s, _, _ in self.env.signals):
                continue
            if any(abs(driver.station - station) < DENSE_SPAWN_CLEARANCE_M + junction.DefaultVehicle.DEFAULT_LENGTH
                   for driver in self.active if driver.kind == "lead"):
                continue
            return float(station)
        return None

    def spawn(self, plan, station, speed, kind, crossing=None, stopping=False):
        self.metrics["spawn_attempts"] += 1
        if station < 4 or station > plan.length_m - 10:
            self.metrics["spawn_deferred"] += 1
            return False
        point = self.geometry(plan).extended_position(station)
        vehicles = [self.env.vehicle] + [driver.vehicle for driver in self.active]
        if any(np.linalg.norm(vehicle.position - point) < DENSE_SPAWN_CLEARANCE_M for vehicle in vehicles):
            self.metrics["spawn_deferred"] += 1
            return False
        half_length = junction.DefaultVehicle.DEFAULT_LENGTH / 2
        for stop_s, tls, link in plan.signals:
            distance = stop_s - station - half_length
            if distance > 0:
                color = junction.libsumo.trafficlight.getRedYellowGreenState(tls)[link].lower()
                if color in ("r", "u", "y"):
                    speed = min(speed, math.sqrt(2 * DENSE_SPAWN_DECEL_MPS2 * max(0., distance - DENSE_SPAWN_BUFFER_M)))
                break
        for driver in self.active:
            if driver.plan is plan and driver.station > station:
                gap = driver.station - station - (driver.vehicle.LENGTH + junction.DefaultVehicle.DEFAULT_LENGTH) / 2
                speed = min(speed, max(0.0, (gap - DENSE_SPAWN_BUFFER_M) / DENSE_INITIAL_HEADWAY_S))
        cruise = float(self.rng.uniform(*city.NPC_CRUISE_MPS))
        driver = DenseNpcDriver(self, plan, station, speed, kind, crossing=crossing, stopping=stopping)
        driver.cruise = cruise
        self.active.append(driver)
        self.spawn_events.append(dict(uid=driver.uid, kind=kind, time=self.time, station=station))
        self.metrics["spawned"] += 1
        self.metrics[f"{kind}_spawned"] += 1
        self.metrics["peak_active"] = max(self.metrics["peak_active"], len(self.active))
        return True

    def before_step(self):
        self.time += route.DRIVE_DT_S
        for item in self.crossings:
            ahead = item["ego_s"] - self.env.station
            if not (8 < ahead < self.settings["activation_m"] and
                    item["spawned"] < self.settings["crossing_count"] and
                    self.time >= item["next_release"] and len(self.active) < self.settings["active_limit"]):
                continue
            station = max(4.0, item["npc_s"] - DENSE_CROSS_APPROACH_M)
            speed = float(self.rng.uniform(*city.NPC_CRUISE_MPS))
            if self.spawn(item["plan"], station, speed, "cross",
                          crossing=(item["ego_s"], item["npc_s"], item["junction_id"])):
                item["spawned"] += 1
                item["next_release"] = self.time + float(self.rng.uniform(*self.settings["release_s"]))
        for driver in self.active:
            driver.vehicle.before_step(driver.command())
        self.steps += 1
        self.active_steps += len(self.active)
        self.metrics["mean_active_npcs"] = self.active_steps / self.steps
        self.metrics["time_with_8plus_npcs_s"] += route.DRIVE_DT_S * (len(self.active) >= 8)


def sample_route(net, rng, endpoints=None, length_range=ROUTE_LENGTH_M, excluded_pairs=()):
    lanes = endpoints if endpoints is not None else route.list_endpoints(net)
    if len(lanes) < 2:
        raise ValueError("Random routing needs two connected passenger lanes.")
    for _ in range(ROUTE_ATTEMPTS):
        start, end = rng.choice(lanes, size=2, replace=False)
        if (start, end) in excluded_pairs:
            continue
        offsets = []
        for lane in (start, end):
            length = route.geomhelper.polyLength(net.getLane(lane).getShape())
            margin = min(ENDPOINT_MARGIN_M, length * 0.2)
            offsets.append(float(rng.uniform(margin, length - margin)))
        try:
            plan = route.plan_route(net, str(start), str(end), start_offset=offsets[0], end_offset=offsets[1])
        except ValueError:
            continue
        if length_range[0] <= plan.length_m <= length_range[1]:
            return plan, dict(start=str(start), end=str(end), start_offset=offsets[0],
                             end_offset=offsets[1], length_m=plan.length_m)
    raise ValueError(f"No connected route in length range {length_range} after {ROUTE_ATTEMPTS} attempts.")


class DenseTrafficEnv(city.MultiTrafficEnv):
    def __init__(self, map_path, *, density="dense", render=False, random_routes=False, traffic="mixed"):
        self.density = density
        self.random_routes = random_routes
        self.training = False
        self.route_history = []
        super().__init__(map_path, render=render, traffic=traffic)
        self.endpoints = route.list_endpoints(self.net)

    def reset(self, seed=None, options=None):
        seed = int(self.seed_rng.integers(*TRAIN_SEEDS)) if seed is None else int(seed)
        if self.training:
            curriculum = np.random.default_rng(seed + 91_037)
            self.traffic = str(curriculum.choice(TRAIN_TRAFFIC_MODES))
            self.density = str(curriculum.choice(TRAIN_DENSITIES))
            if self.population is not None:
                self.population.density = self.density
                self.population.settings = DENSITY_SETTINGS[self.density]
        if self.random_routes:
            rng = np.random.default_rng(seed + 47_891)
            plan, details = sample_route(self.net, rng, self.endpoints)
            self.set_route(plan)
            self.route_history.append(dict(seed=seed, density=self.density, traffic=self.traffic, **details))
        return super().reset(seed=seed, options=options)

    def _get_reset_return(self, info):
        if self.population is None:
            self.population = DensePopulation(self, self.net, density=self.density)
        return super()._get_reset_return(info)


def crossing_target(env):
    state = env.conflict_state
    if state and state["unsafe"] and state["distance"] > CROSS_ENTRY_TOLERANCE_M:
        speed = math.sqrt(2 * city.CONFLICT_BRAKING_MPS2 * max(0., state["distance"] - city.CONFLICT_STOP_BUFFER_M))
        command = CROSS_TARGET_GAIN * (speed - env.vehicle.speed)
        if state["distance"] <= city.CONFLICT_STOP_BUFFER_M + 0.4:
            command = min(command, city.CONFLICT_HOLD_THROTTLE)
        return float(np.clip(command, -1, 1))
    return 1.0


def collect_data(env, model, rng, steps, corrections=False):
    states, targets = [], []
    finished = True
    for i in range(steps):
        if finished:
            obs, _ = env.reset(seed=int(rng.integers(*TRAIN_SEEDS)))
        target = env.teacher_action()
        states.append(obs.copy())
        targets.append(crossing_target(env))
        action, _ = model.predict(obs, deterministic=True)
        if not corrections:
            action[1] = np.clip(target[1] + rng.normal(0, ACTION_NOISE), -1, 1)
        next_obs, reward, ended, truncated, info = env.step(action)
        finished = ended or truncated
        info["TimeLimit.truncated"] = truncated and not ended
        model.replay_buffer.add(obs[None], next_obs[None], action[None], np.array([reward]),
                                np.array([finished]), [info])
        obs = next_obs
        if (i + 1) % 5000 == 0:
            print(f"Collected {i + 1}/{steps} transitions on {len(env.route_history)} random routes.", flush=True)
    return np.stack(states), np.asarray(targets, dtype=np.float32)


def loss_function(actor, states, targets, rng):
    groups = []
    present = states[:, 18] > 0.5
    for mask in (present, present & (targets < -0.2),
                 present & (states[:, 19] < 0), present & (targets > 0.25)):
        if mask.any():
            groups.append(np.flatnonzero(mask))
    if not groups:
        raise ValueError("Training did not encounter traffic.")
    inputs, labels = torch.from_numpy(states), torch.from_numpy(targets)

    def loss():
        indices = np.concatenate([rng.choice(pool, BATCH_SIZE // len(groups)) for pool in groups])
        batch, target = inputs[indices], labels[indices]
        predicted = actor.mu.cross_pedal(batch[:, city.CROSS_INPUTS]).tanh().flatten()
        return torch.nn.functional.mse_loss(predicted, target)

    return loss


def train_dense(env, checkpoint, steps):
    env.training = True
    env.seed_rng = np.random.default_rng(TRAIN_SEED + 1)
    model = SAC.load(checkpoint, env=env, device="cpu", seed=TRAIN_SEED)
    if model.observation_space.shape != (24,) or not isinstance(model.actor.mu, city.CityControls):
        raise ValueError("Use a 24-input city-traffic checkpoint.")
    actor = model.actor
    actor.requires_grad_(False)
    actor.mu.cross_pedal.requires_grad_(True)
    trainable = [p for p in actor.parameters() if p.requires_grad]
    rng = np.random.default_rng(TRAIN_SEED)
    states, targets = collect_data(env, model, rng, DEMO_STEPS)
    clone = torch.optim.Adam(trainable, lr=CLONE_LR)
    for iteration, updates in enumerate((CLONE_UPDATES, CORRECTION_UPDATES)):
        if iteration:
            extra, labels = collect_data(env, model, rng, CORRECTION_STEPS, corrections=True)
            states, targets = np.vstack((states, extra)), np.concatenate((targets, labels))
        loss = loss_function(actor, states, targets, rng)
        for _ in range(updates):
            clone.zero_grad()
            loss().backward()
            clone.step()
        print(f"Imitation round {iteration + 1}: {len(states)} samples, {updates} updates.", flush=True)

    def anchor(optimizer, args, kwargs):
        for group in optimizer.param_groups:
            group["lr"] = ACTOR_LR
        with torch.enable_grad():
            (ANCHOR_WEIGHT * loss()).backward()

    old_steps, old_updates = model.num_timesteps, model._n_updates
    model.get_env().seed(int(rng.integers(*TRAIN_SEEDS)))
    hook = actor.optimizer.register_step_pre_hook(anchor)
    try:
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
    finally:
        hook.remove()
    print(f"SAC steps {old_steps} -> {model.num_timesteps}, updates {old_updates} -> {model._n_updates}.", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--model", type=Path, default=MODEL_FILE)
    parser.add_argument("--density", choices=("moderate", "dense"), default="dense")
    parser.add_argument("--train-from", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=ONLINE_STEPS)
    parser.add_argument("--random", action="store_true", help="Sample a new connected start and destination for each trip.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pick", action="store_true")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--from-xy", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--to-xy", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    endpoints = any(x is not None for x in (args.start, args.end, args.from_xy, args.to_xy))
    if bool(args.train_from) != bool(args.output):
        parser.error("Use --train-from and --output together.")
    if args.steps < 1 or args.episodes < 1 or not 0 <= args.seed <= 2**31 - 1 - args.episodes:
        parser.error("Choose positive steps/episodes and seeds within the simulator range.")
    if (args.random or args.pick or args.train_from) and endpoints:
        parser.error("Choose endpoints or an automatic route selection mode.")
    if args.pick and args.random or args.start and args.from_xy or args.end and args.to_xy:
        parser.error("Choose one input method per endpoint.")
    if args.train_from and (args.render or args.pick):
        parser.error("Training automatically samples random routes.")
    if args.episodes > 1 and not args.random:
        parser.error("Use --random for multiple trips.")
    if not args.map.is_file():
        parser.error(f"Map file does not exist: {args.map}")
    torch.set_num_threads(1)
    env = DenseTrafficEnv(args.map, density=args.density, random_routes=bool(args.train_from or args.random), render=args.render)
    try:
        if args.train_from:
            model = train_dense(env, args.train_from, args.steps)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.output)
            print(f"Saved {args.output}. Used {len(env.route_history)} random route episodes.")
            return 0
        if not args.random:
            if args.pick or not endpoints:
                plan = route.select_on_map(env.net)
                if plan is None:
                    return 0
            else:
                def endpoint(name, xy):
                    if name is not None:
                        return name, None
                    if xy is None:
                        parser.error("Supply both endpoints or use --pick.")
                    lane, offset, _ = route.snap_point(env.net, *xy)
                    return lane, offset

                try:
                    start, low = endpoint(args.start, args.from_xy)
                    end, high = endpoint(args.end, args.to_xy)
                    plan = route.plan_route(env.net, start, end, start_offset=low, end_offset=high)
                except ValueError as exc:
                    parser.error(str(exc))
            env.set_route(plan)
        model = SAC.load(args.model, device="cpu")
        passed = 0
        for seed in range(args.seed, args.seed + args.episodes):
            obs, _ = env.reset(seed=seed)
            print(f"Seed {seed}: {env.plan.lane_ids[0]} -> {env.plan.lane_ids[-1]}, {env.plan.length_m:.0f} m", flush=True)
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, ended, truncated, info = env.step(action)
                if args.render:
                    env.render(text={"Remaining": f"{info['remaining_m']:.0f} m",
                                     "Speed": f"{env.vehicle.speed_km_h:.1f} km/h",
                                     "Other cars": str(len(env.population.active))})
                if ended or truncated:
                    break
            failures = [key for key in ("collision", "yield_violation", "out_of_lane", "wrong_way", "had_overspeed", "red_violation",
                                         "yellow_violation", "following_violation", "headway_violation",
                                         "npc_collision", "npc_red_violation", "npc_yellow_violation", "npc_out_of_lane")
                        if info.get(key)]
            if truncated:
                failures.append("timeout")
            success = bool(info["arrived"] and not failures)
            passed += int(success)
            print(f"{'Arrived' if success else 'Failed'}: goal error {info['endpoint_distance_m']:.2f} m, "
                  f"peak traffic {info['peak_active']}, crossings {info['cross_passages']}, failures {failures}", flush=True)
        print(f"Completed {passed}/{args.episodes} trips.")
        return 0 if passed == args.episodes else 1
    finally:
        env.close()


if __name__ == "__main__":
    from train_sac_dense_traffic import main as run
    raise SystemExit(run())

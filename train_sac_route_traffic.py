"""Train or drive a SAC route agent with a slowing, stopping lead vehicle."""

import argparse
import math
from pathlib import Path

import train_sac_route as route

np, torch, SAC, junction = route.np, route.torch, route.SAC, route.junction
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "models" / "sac_route_traffic.zip"
BASE_MODEL = PROJECT_DIR / "models" / "sac_route.zip"
TRAINING_SEED = 307
TRAINING_SEEDS = (410_000, 610_000)
DEMO_STEPS = 20_000
CORRECTION_STEPS = 6_000
CLONE_UPDATES = 3_000
CORRECTION_UPDATES = 1_000
ONLINE_STEPS = 12_000
BATCH_SIZE = 128
CLONE_LR = 0.001
ACTOR_LR = 0.00004
ANCHOR_WEIGHT = 40.0
DEMO_NOISE = 0.08


TRAFFIC_CASES = ("cruise", "brake", "stop")
INITIAL_GAP_M = (12.0, 17.0)
LEAD_CRUISE_MPS = (3.4, 4.7)
LEAD_BRAKE_MPS2 = 2.4
LEAD_SLOW_MPS = 1.2
LEAD_HOLD_S = (6.0, 8.0)
MIN_FOLLOW_GAP_M = 2.0
FOLLOW_HEADWAY_S = 2.0
HEADWAY_GRACE_S = 1.0
FOLLOW_STANDSTILL_M = 4.0
FOLLOW_STOP_ZONE_M = 10.0
FOLLOW_STOP_SPEED_MPS = 0.20
FOLLOW_DWELL_S = 0.6
FOLLOW_GAP_GAIN = 0.55
FOLLOW_SPEED_GAIN = 0.8
OBS_GAP_M = 30.0
OBS_SPEED_MPS = 10.0
MIN_TRAFFIC_ROUTE_M = 45.0
LEAD_EVENT_OFFSET_M = (7.0, 12.0)
LEAD_SIGNAL_BUFFER_M = 2.5
LEAD_SIGNAL_BRAKE_MPS2 = 1.8
LEAD_TARGET_SPEED_RATIO = 0.8
LEAD_SPEED_GAIN = 0.65
LEAD_TIGHT_TURN_GAIN = 1.02
FOLLOW_REWARD_COST = 0.22
FAILURE_REWARD = -50.0
BODY_BOUNDARY_TOLERANCE_M = 0.005


class TrafficObservation(route.DriveObservation):
    @property
    def observation_space(self):
        return junction.gym.spaces.Box(-1, 1, (18,), np.float32)

    def observe(self, vehicle):
        env = self.env
        extra = np.zeros(4, dtype=np.float32)
        if env.lead is not None:
            gap, speed, ego = env.lead_state()
            extra[:] = (1.0, gap / OBS_GAP_M, speed / OBS_SPEED_MPS, (ego - speed) / OBS_SPEED_MPS)
        return np.r_[env.drive_obs, np.clip(extra, -1, 1)].astype(np.float32)


class TrafficRouteEnv(route.RouteEnv):
    """A learned driver and one scripted, collidable lead car share a route."""

    def __init__(self, map_path, *, render=False, traffic="mixed"):
        if traffic not in ("mixed", "none", *TRAFFIC_CASES):
            raise ValueError(f"Unknown traffic case: {traffic}")
        self.traffic = traffic
        self.lead = None
        self.lead_s = self.lead_speed = 0.0
        super().__init__(map_path, render=render)

    def get_single_observation(self):
        return TrafficObservation(self.config)

    def reset(self, seed=None, options=None):
        self.remove_lead()
        seed = int(self.seed_rng.integers(1, 2**30)) if seed is None else int(seed)
        self.traffic_rng = np.random.default_rng(seed + 947)
        self.traffic_case = TRAFFIC_CASES[seed % 3] if self.traffic == "mixed" else self.traffic
        self.min_gap_m = self.min_headway_s = self.min_clearance_m = math.inf
        self.headway_duration = self.stop_dwell = 0.0
        self.following_violation = self.headway_violation = False
        self.lead_stopped = self.stopped_behind = self.restarted_behind = False
        self.lead_red_wait_steps = self.lead_signals_crossed = 0
        self.lead_red_violation = self.lead_collision = False
        self.lead_out_of_lane = False
        self.max_lead_boundary_excess_m = 0.0
        return super().reset(seed=seed, options=options)

    def _get_reset_return(self, info):
        if self.traffic_case != "none" and self.length > MIN_TRAFFIC_ROUTE_M:
            rng = self.traffic_rng
            self.lead_s = float(rng.uniform(*INITIAL_GAP_M) + junction.DefaultVehicle.DEFAULT_LENGTH)
            self.lead_speed = float(rng.uniform(*LEAD_CRUISE_MPS))
            self.lead_cruise = self.lead_speed
            self.event_s = min(self.length * 0.5, self.lead_s + rng.uniform(*LEAD_EVENT_OFFSET_M))
            self.event_elapsed = 0.0
            self.event_hold = float(rng.uniform(*LEAD_HOLD_S))
            self.event_complete = self.traffic_case == "cruise"
            self.lead_signal_index = 0
            self.lead_previous_color = ""
            self.lead_clear_yellow = False
            point, heading = self.lead_pose(self.lead_s)
            self.lead = self.engine.map_manager.spawn_object(
                junction.DefaultVehicle, position=tuple(point), heading=heading,
                vehicle_config=dict(navigation_module=None, show_navi_mark=False, show_dest_mark=False))
            self.lead.set_velocity(np.array([math.cos(heading), math.sin(heading)]) * self.lead_speed)
            while (self.lead_signal_index < len(self.signals) and
                   self.signals[self.lead_signal_index][0] <= self.lead_s + self.lead.LENGTH / 2):
                self.lead_signal_index += 1
        else:
            self.traffic_case = "none"
        result = super()._get_reset_return(info)
        if self.lead is not None:
            gap, _, ego = self.lead_state()
            self.min_gap_m = gap
            self.min_clearance_m = rectangle_clearance(self.vehicle, self.lead)
            if ego >= 1.0:
                self.min_headway_s = gap / ego
        return result

    def lead_pose(self, station):
        if station > self.length:
            point = self.points[-1] + self.directions[-1] * (station - self.length)
            tangent = self.directions[-1]
        else:
            point = self.position_at(station)
            tangent = self.position_at(min(self.length, station + 0.15)) - self.position_at(max(0, station - 0.15))
        return point, math.atan2(tangent[1], tangent[0])

    def lead_state(self):
        gap = self.lead_s - self.station - (self.lead.LENGTH + self.vehicle.LENGTH) / 2
        ego = max(0.0, float(self.vehicle.velocity @ self.directions[self.segment_idx]))
        return float(gap), float(self.lead_speed), ego

    def lead_command(self):
        target = self.lead_cruise
        i = self.segment(self.lead_s)
        a, b = self.segment(self.lead_s - 5), self.segment(self.lead_s + 12) + 1
        curve = np.max(np.abs(self.directions[a:b, 0] * self.directions[i, 1] -
                              self.directions[a:b, 1] * self.directions[i, 0]))
        target = min(target, LEAD_TARGET_SPEED_RATIO * min(self.plan.segment_speeds[a:b]),
                     route.MAX_CRUISE_MPS - (route.MAX_CRUISE_MPS - route.CORNER_SPEED_MPS) * min(1, 2 * curve))
        if not self.event_complete and self.lead_s >= self.event_s:
            slow = 0.0 if self.traffic_case == "stop" else LEAD_SLOW_MPS
            target = min(target, slow)
            if self.lead_speed <= slow + FOLLOW_STOP_SPEED_MPS:
                self.event_elapsed += route.DRIVE_DT_S
            if self.event_elapsed >= self.event_hold:
                self.event_complete = True
        distance, color = math.inf, "g"
        if self.lead_signal_index < len(self.signals):
            stop_s, tls, link = self.signals[self.lead_signal_index]
            distance = stop_s - self.lead_s - self.lead.LENGTH / 2
            color = junction.libsumo.trafficlight.getRedYellowGreenState(tls)[link].lower()
            if color != "y":
                self.lead_clear_yellow = False
            elif self.lead_previous_color != "y":
                self.lead_clear_yellow = distance < self.lead_speed**2 / (2 * LEAD_BRAKE_MPS2) + self.lead_speed * route.DRIVE_DT_S + 0.5
            stopping = color in ("r", "u") or color == "y" and not self.lead_clear_yellow
            if stopping:
                target = min(target, math.sqrt(2 * LEAD_SIGNAL_BRAKE_MPS2 * max(0, distance - LEAD_SIGNAL_BUFFER_M)))
            self.lead_red_wait_steps += int(color in ("r", "u") and distance < 8 and self.lead_speed < 0.2)
        self.lead_previous_color = color
        point, _ = self.lead_pose(self.lead_s + route.PREVIEW_M[0])
        delta = point - self.lead.position
        bearing = math.atan2(delta[1], delta[0]) - self.lead.heading_theta
        preview = math.sin(bearing)
        if len(self.bend_stations) and np.min(abs(self.bend_stations - self.lead_s)) < 10:
            preview *= LEAD_TIGHT_TURN_GAIN
        wheelbase = self.lead.FRONT_WHEELBASE + self.lead.REAR_WHEELBASE
        steer = math.atan(2 * wheelbase * preview / route.PREVIEW_M[0]) / math.radians(self.lead.max_steering)
        throttle = LEAD_SPEED_GAIN * (target - self.lead.speed)
        if target <= 0.01:
            throttle = min(throttle, -0.3)
        return np.clip([steer, throttle], -1, 1), distance, color

    def _step_simulator(self, actions):
        old_s = self.lead_s
        if self.lead is not None:
            action, old_distance, old_color = self.lead_command()
            self.lead.before_step(action)
        result = super()._step_simulator(actions)
        if self.lead is not None:
            self.lead.after_step()
            self.lead_s, _, idx = self.project(self.lead.position, old_s)
            if self.lead_s >= self.length - 0.01:
                self.lead_s = self.length + max(0, float((self.lead.position - self.points[-1]) @ self.directions[-1]))
            self.lead_speed = max(0.0, float(self.lead.velocity @ self.directions[idx]))
            if old_distance > 0 >= old_distance - (self.lead_s - old_s):
                self.lead_red_violation |= old_color in ("r", "u")
                self.lead_signals_crossed += 1
                self.lead_signal_index += 1
                self.lead_previous_color = ""
            self.score_following()
            if self.lead_s >= self.length + self.lead.LENGTH:
                self.remove_lead()
        self.step_info.update(
            traffic_case=self.traffic_case, min_gap_m=self.min_gap_m if math.isfinite(self.min_gap_m) else None,
            min_time_headway_s=self.min_headway_s if math.isfinite(self.min_headway_s) else None,
            min_clearance_m=self.min_clearance_m if math.isfinite(self.min_clearance_m) else None,
            following_violation=self.following_violation, headway_violation=self.headway_violation,
            lead_stopped=self.lead_stopped, stopped_behind=self.stopped_behind, restarted_behind=self.restarted_behind,
            lead_red_violation=self.lead_red_violation, lead_red_wait_steps=self.lead_red_wait_steps,
            lead_signals_crossed=self.lead_signals_crossed, lead_collision=self.lead_collision,
            lead_out_of_lane=self.lead_out_of_lane, max_lead_boundary_excess_m=self.max_lead_boundary_excess_m)
        return result

    def score_following(self):
        gap, speed, ego = self.lead_state()
        self.min_gap_m = min(self.min_gap_m, gap)
        if ego >= 1.0:
            self.min_headway_s = min(self.min_headway_s, gap / ego)
        self.headway_duration = self.headway_duration + route.DRIVE_DT_S if ego >= 1 and gap < FOLLOW_HEADWAY_S * ego else 0
        self.headway_violation |= self.headway_duration + 1e-6 >= HEADWAY_GRACE_S
        self.following_violation |= gap < MIN_FOLLOW_GAP_M
        self.lead_stopped |= speed <= FOLLOW_STOP_SPEED_MPS
        stopped = speed <= FOLLOW_STOP_SPEED_MPS and self.vehicle.speed <= FOLLOW_STOP_SPEED_MPS and MIN_FOLLOW_GAP_M <= gap <= FOLLOW_STOP_ZONE_M
        self.stop_dwell = self.stop_dwell + route.DRIVE_DT_S if stopped else 0
        self.stopped_behind |= self.stop_dwell + 1e-6 >= FOLLOW_DWELL_S
        self.restarted_behind |= self.stopped_behind and min(speed, ego) >= 1
        clearance = rectangle_clearance(self.vehicle, self.lead)
        self.lead_collision |= bool(self.lead.crash_vehicle or self.vehicle.crash_vehicle or clearance <= 0)
        self.min_clearance_m = min(self.min_clearance_m, clearance)
        if self.lead.LENGTH < self.lead_s < self.length - self.lead.LENGTH:
            excess = max(abs(lateral) - self.plan.segment_widths[index] / 2
                         for _, lateral, index in (self.project(corner, self.lead_s)
                                                   for corner in junction.vehicle_corners(self.lead)))
            self.max_lead_boundary_excess_m = max(self.max_lead_boundary_excess_m, float(excess))
            self.lead_out_of_lane |= excess > BODY_BOUNDARY_TOLERANCE_M
        self.step_info["reward"] -= FOLLOW_REWARD_COST * route.DRIVE_DT_S * max(0, FOLLOW_HEADWAY_S * ego - gap)
        if self.following_violation:
            self.step_info.update(failed=True, arrived=False, reward=FAILURE_REWARD)
        if self.lead_collision:
            self.step_info.update(collision=True, failed=True, arrived=False, reward=FAILURE_REWARD)

    def teacher_action(self):
        action = super().teacher_action()
        if self.lead is not None:
            gap, speed, ego = self.lead_state()
            target = speed + FOLLOW_GAP_GAIN * (gap - FOLLOW_STANDSTILL_M - FOLLOW_HEADWAY_S * ego)
            if speed < FOLLOW_STOP_SPEED_MPS and gap <= FOLLOW_STOP_ZONE_M - 0.4:
                target = 0
            target = max(0, min(target, self.target_speed))
            action[1] = min(action[1], np.clip(FOLLOW_SPEED_GAIN * (target - ego), -1, 1))
        return action

    def remove_lead(self):
        if self.lead is not None:
            self.engine.map_manager.clear_objects([self.lead.id], force_destroy=True)
            self.lead = None

    def close(self):
        self.remove_lead()
        super().close()


def rectangle_clearance(a, b):
    rectangles = []
    for car in (a, b):
        forward = np.asarray(car.heading) * car.LENGTH / 2
        side = np.array([-car.heading[1], car.heading[0]]) * car.WIDTH / 2
        rectangles.append(np.array([car.position + forward + side, car.position + forward - side,
                                    car.position - forward - side, car.position - forward + side]))
    separated = False
    closest = math.inf
    for polygon, other in (rectangles, rectangles[::-1]):
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = end - start
            normal = np.array([-edge[1], edge[0]])
            p, q = polygon @ normal, other @ normal
            separated |= p.max() < q.min() or q.max() < p.min()
            t = np.clip((other - start) @ edge / (edge @ edge), 0, 1)
            closest = min(closest, float(np.linalg.norm(other - start - t[:, None] * edge, axis=1).min()))
    return closest if separated else 0.0


class TrafficControls(route.RouteControls):
    def __init__(self):
        super().__init__()
        self.traffic_pedal = torch.nn.Sequential(
            torch.nn.Linear(11, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1),
        )

    def forward(self, obs):
        original = super().forward(obs[:, :14])
        inputs = obs[:, [2, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17]]
        following = torch.minimum(self.traffic_pedal(inputs), original[:, 1:])
        pedal = torch.where(obs[:, 14:15] > 0.5, following, original[:, 1:])
        return torch.cat((original[:, :1], pedal), dim=1)


class TrafficPolicy(route.junction.SACPolicy):
    def make_actor(self, features_extractor=None):
        actor = super().make_actor(features_extractor)
        actor.mu = TrafficControls().to(self.device)
        actor.requires_grad_(False)
        actor.mu.traffic_pedal.requires_grad_(True)
        return actor


def extend_traffic_parameter(name, value, shape):
    if value.shape == shape:
        return value.clone()
    if value.ndim != 2 or shape != (value.shape[0], value.shape[1] + 4):
        raise ValueError(f"Unexpected checkpoint parameter {name}: {value.shape} -> {shape}.")
    zeros = value.new_zeros(value.shape[0], 4)
    if name == "log_std.weight":
        return torch.cat((value, zeros), dim=1)
    # Critic inputs put the two actions after the observation.
    return torch.cat((value[:, :-2], zeros, value[:, -2:]), dim=1)


def copy_traffic_optimizer(source, target):
    targets = dict(target.named_parameters())
    for name, parameter in source.named_parameters():
        destination = targets[name]
        for key, value in source.optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor):
                value = extend_traffic_parameter(name, value, destination.shape) if value.ndim else value.clone()
            target.optimizer.state[destination][key] = value
    for old, new in zip(source.optimizer.param_groups, target.optimizer.param_groups):
        new.update({key: value for key, value in old.items() if key != "params"})


def load_traffic_model(checkpoint, env, seed):
    model = route.SAC.load(checkpoint, device="cpu", seed=seed)
    if env.observation_space.shape != (18,):
        raise ValueError("Traffic driving requires an 18-input environment.")
    if model.observation_space.shape == (18,) and isinstance(model.actor.mu, TrafficControls):
        model.set_env(env)
        return model
    if model.observation_space.shape != (14,) or not isinstance(model.actor.mu, route.RouteControls):
        raise ValueError("Start from the 14-input route model or an 18-input traffic model.")
    previous = model.policy
    entropy = model.log_ent_coef.detach().clone()
    entropy_optimizer = model.ent_coef_optimizer.state_dict()
    model.observation_space = env.observation_space
    model.set_env(env)
    model.policy_class = TrafficPolicy
    model.replay_buffer = None
    model._setup_model()
    actor_state = model.actor.state_dict()
    for name, value in previous.actor.state_dict().items():
        actor_state[name] = extend_traffic_parameter(name, value, actor_state[name].shape)
    for name, value in previous.actor.mu.pedal.state_dict().items():
        if name == "0.weight":
            value = torch.cat((value, value.new_zeros(value.shape[0], 3)), dim=1)
        actor_state[f"mu.traffic_pedal.{name}"] = value.clone()
    model.actor.load_state_dict(actor_state)
    for source, target in ((previous.critic, model.critic), (previous.critic_target, model.critic_target)):
        state = target.state_dict()
        target.load_state_dict({name: extend_traffic_parameter(name, value, state[name].shape)
                                for name, value in source.state_dict().items()})
    copy_traffic_optimizer(previous.actor, model.actor)
    copy_traffic_optimizer(previous.critic, model.critic)
    with torch.no_grad():
        model.log_ent_coef.copy_(entropy)
    model.ent_coef_optimizer.load_state_dict(entropy_optimizer)
    return model


def collect_demonstrations(env, model, rng, count, corrections=False):
    states, targets = [], []
    done = True
    for _ in range(count):
        if done:
            obs, _ = env.reset(seed=int(rng.integers(*TRAINING_SEEDS)))
        target = env.teacher_action()
        states.append(obs.copy())
        targets.append(target[1])
        action, _ = model.predict(obs, deterministic=True)
        if not corrections:
            action[1] = np.clip(target[1] + rng.normal(0, DEMO_NOISE), -1, 1)
        next_obs, reward, ended, truncated, info = env.step(action)
        done = ended or truncated
        info["TimeLimit.truncated"] = truncated and not ended
        model.replay_buffer.add(obs[None], next_obs[None], action[None], np.array([reward]),
                                np.array([done]), [info])
        obs = next_obs
    return np.stack(states), np.asarray(targets, dtype=np.float32)


def imitation_loss(actor, states, targets, rng):
    groups = [np.flatnonzero(states[:, 14] > 0.5)]
    for mask in ((states[:, 14] > 0.5) & (states[:, 16] < 0.03),
                 (states[:, 14] > 0.5) & (states[:, 17] > 0.04),
                 (states[:, 14] > 0.5) & (states[:, 17] < -0.04)):
        if mask.any():
            groups.append(np.flatnonzero(mask))
    if not len(groups[0]):
        raise ValueError("Training needs a lead vehicle.")
    states, targets = torch.from_numpy(states), torch.from_numpy(targets)

    def loss():
        per_group = BATCH_SIZE // (2 * len(groups))
        indices = np.concatenate([rng.choice(g, per_group) for g in groups[1:]] +
                                 [rng.choice(groups[0], BATCH_SIZE - per_group * (len(groups) - 1))])
        return torch.nn.functional.mse_loss(actor(states[indices], deterministic=True)[:, 1], targets[indices])

    return loss


def train_traffic(env, checkpoint, steps):
    model = load_traffic_model(checkpoint, env, TRAINING_SEED)
    model.actor.requires_grad_(False)
    model.actor.mu.traffic_pedal.requires_grad_(True)
    rng = np.random.default_rng(TRAINING_SEED)
    states, labels = collect_demonstrations(env, model, rng, DEMO_STEPS)
    clone = torch.optim.Adam(model.actor.mu.traffic_pedal.parameters(), lr=CLONE_LR)
    for round_index, updates in enumerate((CLONE_UPDATES, CORRECTION_UPDATES)):
        if round_index:
            extra, targets = collect_demonstrations(env, model, rng, CORRECTION_STEPS, corrections=True)
            states, labels = np.vstack((states, extra)), np.concatenate((labels, targets))
        loss = imitation_loss(model.actor, states, labels, rng)
        for _ in range(updates):
            clone.zero_grad()
            loss().backward()
            clone.step()
        print(f"Imitation round {round_index + 1}: {len(states)} physical transitions, {updates} updates.", flush=True)

    def anchor(optimizer, args, kwargs):
        for group in optimizer.param_groups:
            group["lr"] = ACTOR_LR
        with torch.enable_grad():
            (ANCHOR_WEIGHT * loss()).backward()

    previous_steps, previous_updates = model.num_timesteps, model._n_updates
    model.get_env().seed(int(rng.integers(*TRAINING_SEEDS)))
    hook = model.actor.optimizer.register_step_pre_hook(anchor)
    try:
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
    finally:
        hook.remove()
    print(f"SAC steps {previous_steps} -> {model.num_timesteps}; updates {previous_updates} -> {model._n_updates}.", flush=True)
    return model


def drive(env, plan, model, seed, render):
    env.set_route(plan)
    obs, _ = env.reset(seed=seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, ended, truncated, info = env.step(action)
        if render:
            status = {"Remaining": f"{info['remaining_m']:.0f} m", "Speed": f"{env.vehicle.speed_km_h:.1f} km/h"}
            if env.lead is not None:
                status["Lead speed"] = f"{env.lead_speed * 3.6:.1f} km/h"
            env.render(text=status)
        if ended or truncated:
            break
    stopped_and_resumed = info["traffic_case"] != "stop" or (info["stopped_behind"] and info["restarted_behind"])
    success = (info["arrived"] and not truncated and not info["following_violation"] and not info["headway_violation"]
               and not info["lead_collision"] and not info["lead_red_violation"]
               and not info["lead_out_of_lane"] and stopped_and_resumed)
    print("Arrived" if success else "Trip failed", flush=True)
    for name in ("trip_m", "endpoint_distance_m", "terminal_speed_mps", "signals_crossed",
                 "min_gap_m", "min_time_headway_s", "following_violation", "headway_violation",
                 "stopped_behind", "restarted_behind", "collision"):
        print(f"{name}: {info.get(name)}")
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, help="SUMO .net.xml file.")
    parser.add_argument("--model", type=Path, default=MODEL_FILE)
    parser.add_argument("--train-from", type=Path, help="Resume from a route or route-traffic SAC checkpoint.")
    parser.add_argument("--output", type=Path, help="Save the trained model to this file.")
    parser.add_argument("--steps", type=int, default=ONLINE_STEPS)
    parser.add_argument("--traffic", choices=("mixed", "cruise", "brake", "stop", "none"), default="mixed")
    parser.add_argument("--pick", action="store_true", help="Click two points on the map, then press Enter.")
    parser.add_argument("--start", help="Starting edge or lane ID.")
    parser.add_argument("--end", help="Destination edge or lane ID.")
    parser.add_argument("--from-xy", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--to-xy", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.map = args.map or (route.TRAIN_MAP if args.train_from else route.EXAMPLE_MAP)
    endpoints = any(x is not None for x in (args.start, args.end, args.from_xy, args.to_xy))
    if bool(args.train_from) != bool(args.output):
        parser.error("Use --train-from and --output together.")
    if args.steps < 1 or not 0 <= args.seed < 2**31 - 1:
        parser.error("Choose a positive step count and a seed from 0 to 2147483646.")
    if args.train_from and (args.pick or endpoints or args.render or args.traffic == "none"):
        parser.error("Training requires traffic, and chooses routes automatically.")
    if args.pick and endpoints:
        parser.error("Use --pick or explicit endpoints.")
    if args.start is not None and args.from_xy is not None or args.end is not None and args.to_xy is not None:
        parser.error("Specify each endpoint by ID or coordinates.")
    if not args.map.is_file():
        parser.error(f"Map file does not exist: {args.map}")
    torch.set_num_threads(1)
    net = route.read_network(args.map)
    if args.train_from:
        env = TrafficRouteEnv(args.map, traffic=args.traffic)
        env.set_routes(route.training_routes(net))
        try:
            model = train_traffic(env, args.train_from, args.steps)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.output)
            print(f"Saved {args.output}", flush=True)
        finally:
            env.close()
        return 0
    if args.pick or not endpoints:
        plan = route.select_on_map(net)
        if plan is None:
            return 0
    else:
        def endpoint(name, xy):
            if name is not None:
                return name, None
            if xy is None:
                parser.error("Supply both endpoints, or use --pick.")
            lane, offset, _ = route.snap_point(net, *xy)
            return lane, offset

        try:
            start, low = endpoint(args.start, args.from_xy)
            end, high = endpoint(args.end, args.to_xy)
            plan = route.plan_route(net, start, end, start_offset=low, end_offset=high)
        except ValueError as exc:
            parser.error(str(exc))
    print("Route: " + " -> ".join(plan.edge_ids), flush=True)
    model = SAC.load(args.model, device="cpu")
    env = TrafficRouteEnv(args.map, traffic=args.traffic, render=args.render)
    try:
        return 0 if drive(env, plan, model, args.seed, args.render) else 1
    finally:
        env.close()


if __name__ == "__main__":
    from train_sac_route_traffic import main as run
    raise SystemExit(run())

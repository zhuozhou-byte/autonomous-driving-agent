"""Drive a planned SUMO route through queues and crossing traffic with SAC."""

import argparse
import math
from pathlib import Path

import train_sac_route as route
import train_sac_route_traffic as traffic

np, torch, SAC, junction = route.np, route.torch, route.SAC, route.junction
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "models" / "sac_city_traffic.zip"
TRAIN_SEED = 419
TRAIN_SEEDS = (310_000, 510_000)
DEMO_STEPS = 28_000
CORRECTION_STEPS = 8_000
CLONE_UPDATES = 3_500
CORRECTION_UPDATES = 1_500
ONLINE_STEPS = 16_000
BATCH_SIZE = 128
CLONE_LR = 0.001
ACTOR_LR = 0.000025
ANCHOR_WEIGHT = 50.0
ACTION_NOISE = 0.06
CROSS_INPUTS = (2, 6, 7, 19, 20, 21, 22, 23)


NPC_LEAD_GAPS_M = (17.0, 38.0)
NPC_CRUISE_MPS = (3.6, 4.8)
NPC_MAX_ACTIVE = 6
CROSS_ACTIVATION_M = 38.0
CROSS_SPAWN_DISTANCE_M = (18.0, 45.0)
CROSS_RELEASE_GAP_S = (3.5, 5.0)
NPC_HEADWAY_S = 2.0
NPC_STOP_GAP_M = 4.5
NPC_CROSS_BUFFER_M = 5.0
NPC_OCCUPIED_HALF_LENGTH_M = 4.8
NPC_BODY_TOLERANCE_M = 0.005


class NpcGeometry:
    segment = route.RouteEnv.segment
    position_at = route.RouteEnv.position_at
    project = route.RouteEnv.project

    def __init__(self, plan):
        self.plan = plan
        self.points = np.asarray(plan.points, dtype=float)
        vectors = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(vectors, axis=1)
        self.directions = vectors / self.lengths[:, None]
        self.stations = np.r_[0., self.lengths.cumsum()]
        self.length = float(self.stations[-1])
        angles = np.unwrap(np.arctan2(self.directions[:, 1], self.directions[:, 0]))
        curvature = np.diff(angles) / (0.5 * (self.lengths[:-1] + self.lengths[1:]))
        narrow = np.minimum(plan.segment_widths[:-1], plan.segment_widths[1:]) < route.NARROW_LANE_M
        sharp = np.flatnonzero(abs(curvature) > np.where(narrow, route.NARROW_BEND_CURVATURE, route.TIGHT_BEND_CURVATURE))
        separated = ((np.diff(self.stations[sharp + 1]) > 7.) |
                     (np.diff(np.sign(curvature[sharp])) != 0))
        bends = [g for g in np.split(sharp, np.flatnonzero(separated) + 1) if len(g)]
        self.bend_stations = np.array([self.stations[g + 1].mean() for g in bends])
        self.bend_signs = np.array([np.sign(curvature[g[0]]) for g in bends])
        self.bend_offsets = [route.NARROW_BEND_OFFSETS_M if narrow[g].all() and
                             max(abs(curvature[g])) <= route.TIGHT_BEND_CURVATURE else route.BEND_OUTWARD_OFFSETS_M
                             for g in bends]

    def extended_position(self, station):
        if station > self.length:
            return self.points[-1] + self.directions[-1] * (station - self.length)
        return self.position_at(station)


class NpcDriver:
    def __init__(self, population, plan, station, speed, kind, *, crossing=None, stopping=False):
        self.population = population
        self.env = population.env
        self.plan = plan
        self.geometry = population.geometry(plan)
        self.station, self.speed, self.cruise = station, speed, speed
        self.kind = self.role = kind
        self.ego_cross_s, self.cross_s, self.junction_id = crossing or (None, None, None)
        self.signal_index = self.signals_crossed = self.red_wait_steps = 0
        self.previous_color, self.clear_yellow = "", False
        self.red_violation = self.yellow_violation = self.collision = self.out_of_lane = False
        self.max_boundary_excess_m = 0.0
        self.stop_at = station + 35.0 if stopping else math.inf
        self.stop_elapsed = 0.0
        self.stop_hold = float(population.rng.uniform(4.0, 6.0))
        self.last_distance, self.last_color = math.inf, "g"
        self.crossed = False
        geom = self.geometry
        point = geom.extended_position(station)
        tangent = geom.extended_position(station + 0.15) - geom.extended_position(station - 0.15)
        heading = math.atan2(tangent[1], tangent[0])
        self.vehicle = self.env.engine.map_manager.spawn_object(
            junction.DefaultVehicle, position=tuple(point), heading=heading,
            vehicle_config=dict(navigation_module=None, show_navi_mark=False, show_dest_mark=False))
        self.uid = self.vehicle.id
        self.radius = math.hypot(self.vehicle.LENGTH, self.vehicle.WIDTH) / 2
        self.vehicle.set_velocity(np.array([math.cos(heading), math.sin(heading)]) * speed)
        while self.signal_index < len(plan.signals) and plan.signals[self.signal_index][0] <= station + self.vehicle.LENGTH / 2:
            self.signal_index += 1

    def should_yield_to_cross(self, other):
        return -5.0 < other.cross_s - other.station < 23.0

    def command(self):
        geom, car = self.geometry, self.vehicle
        i = geom.segment(self.station)
        a, b = geom.segment(self.station - 5), geom.segment(self.station + 12) + 1
        curve = np.max(abs(geom.directions[a:b, 0] * geom.directions[i, 1] -
                           geom.directions[a:b, 1] * geom.directions[i, 0]))
        target = min(self.cruise, 0.8 * min(self.plan.segment_speeds[a:b]),
                     route.MAX_CRUISE_MPS - (route.MAX_CRUISE_MPS - route.CORNER_SPEED_MPS) * min(1, 2 * curve))
        if self.station >= self.stop_at and self.stop_elapsed < self.stop_hold:
            target = 0.0
            self.stop_elapsed += route.DRIVE_DT_S if self.speed < 0.2 else 0.0
        self.last_distance, self.last_color = math.inf, "g"
        if self.signal_index < len(self.plan.signals):
            s, tls, link = self.plan.signals[self.signal_index]
            self.last_distance = s - self.station - car.LENGTH / 2
            self.last_color = junction.libsumo.trafficlight.getRedYellowGreenState(tls)[link].lower()
            if self.last_color != "y":
                self.clear_yellow = False
            elif self.previous_color != "y":
                self.clear_yellow = self.last_distance < self.speed**2 / (2 * 2.4) + self.speed * route.DRIVE_DT_S + 0.5
            stop = self.last_color in ("r", "u") or self.last_color == "y" and not self.clear_yellow
            if stop:
                target = min(target, math.sqrt(2 * 1.8 * max(0, self.last_distance - 2.5)))
            self.red_wait_steps += int(self.last_color in ("r", "u") and self.last_distance < 8 and self.speed < 0.2)
        self.previous_color = self.last_color
        for other in self.population.active:
            if other is self:
                continue
            if other.plan is self.plan and other.station > self.station:
                gap = other.station - self.station - (car.LENGTH + other.vehicle.LENGTH) / 2
                follow = other.speed + 0.55 * (gap - NPC_STOP_GAP_M - NPC_HEADWAY_S * self.speed)
                if other.speed < 0.2 and gap < 9.6:
                    follow = 0
                target = min(target, max(0.0, follow))
            elif self.kind == "lead" and other.kind == "cross":
                distance = other.ego_cross_s - self.station - car.LENGTH / 2
                arriving = self.should_yield_to_cross(other)
                if 0 < distance < 45 and arriving:
                    target = min(target, math.sqrt(2 * 1.8 * max(0., distance - NPC_CROSS_BUFFER_M)))
            elif self.kind == "cross" and other.kind == "lead":
                occupied = abs(other.station - self.ego_cross_s) < NPC_OCCUPIED_HALF_LENGTH_M
                distance = self.cross_s - self.station - car.LENGTH / 2
                if occupied and distance > 0:
                    target = min(target, math.sqrt(2 * 1.8 * max(0., distance - NPC_CROSS_BUFFER_M)))
        delta = geom.extended_position(self.station + route.PREVIEW_M[0]) - car.position
        preview = math.sin(math.atan2(delta[1], delta[0]) - car.heading_theta)
        if len(geom.bend_stations) and min(abs(geom.bend_stations - self.station)) < 10:
            preview *= traffic.LEAD_TIGHT_TURN_GAIN
        wheelbase = car.FRONT_WHEELBASE + car.REAR_WHEELBASE
        steer = math.atan(2 * wheelbase * preview / route.PREVIEW_M[0]) / math.radians(car.max_steering)
        throttle = 0.65 * (target - car.speed)
        if target <= 0.01:
            throttle = min(throttle, -0.3)
        return np.clip([steer, throttle], -1, 1)

    def after_step(self):
        old_station = self.station
        self.vehicle.after_step()
        self.station, _, i = self.geometry.project(self.vehicle.position, self.station)
        if self.station >= self.geometry.length - 0.01:
            self.station = self.geometry.length + max(0, float((self.vehicle.position - self.geometry.points[-1]) @ self.geometry.directions[-1]))
        self.speed = max(0., float(self.vehicle.velocity @ self.geometry.directions[i]))
        if self.last_distance > 0 >= self.last_distance - (self.station - old_station):
            self.red_violation |= self.last_color in ("r", "u")
            self.yellow_violation |= self.last_color == "y" and not self.clear_yellow
            self.signals_crossed += 1
            self.signal_index += 1
            self.previous_color = ""
        self.collision |= bool(self.vehicle.crash_vehicle)
        if self.vehicle.LENGTH < self.station < self.geometry.length - self.vehicle.LENGTH:
            excess = max(abs(lat) - self.plan.segment_widths[i] / 2
                         for _, lat, i in (self.geometry.project(p, self.station) for p in junction.vehicle_corners(self.vehicle)))
            self.max_boundary_excess_m = max(self.max_boundary_excess_m, float(excess))
            self.out_of_lane |= excess > NPC_BODY_TOLERANCE_M
        if self.kind == "cross" and old_station <= self.cross_s < self.station:
            self.crossed = True

    def finished_route(self):
        return self.station > self.geometry.length + self.vehicle.LENGTH

    def close(self):
        self.env.engine.map_manager.clear_objects([self.vehicle.id], force_destroy=True)


def path_crossings(a, b):
    av, bv = np.diff(a.points, axis=0), np.diff(b.points, axis=0)
    al, bl = np.linalg.norm(av, axis=1), np.linalg.norm(bv, axis=1)
    ast, bst = np.r_[0, al.cumsum()], np.r_[0, bl.cumsum()]
    for i, u in enumerate(av):
        delta = b.points[:-1] - a.points[i]
        denominator = u[0] * bv[:, 1] - u[1] * bv[:, 0]
        good = abs(denominator) > 0.45 * al[i] * bl
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (delta[:, 0] * bv[:, 1] - delta[:, 1] * bv[:, 0]) / denominator
            q = (delta[:, 0] * u[1] - delta[:, 1] * u[0]) / denominator
        for j in np.flatnonzero(good & (t >= 0) & (t <= 1) & (q >= 0) & (q <= 1)):
            yield float(ast[i] + t[j] * al[i]), float(bst[j] + q[j] * bl[j])


class TrafficPopulation:
    def __init__(self, env, net):
        self.env, self.net = env, net
        self.active = []
        self.spawn_events, self.remove_events = [], []
        self._geometry = {}

    @property
    def drivers(self):
        return self.active

    def geometry(self, plan):
        key = id(plan)
        if key not in self._geometry:
            self._geometry[key] = NpcGeometry(plan)
        return self._geometry[key]

    def reset(self, seed, mode):
        self.close()
        self._geometry = {}
        self.rng = np.random.default_rng(seed + 317)
        self.mode = mode
        self.time = 0.0
        self.spawn_events, self.remove_events = [], []
        self.metrics = dict(spawned=0, lead_spawned=0, cross_spawned=0, peak_active=0, cross_passages=0,
                            npc_collision=False, npc_red_violation=False, npc_yellow_violation=False, npc_out_of_lane=False)
        self.crossings = []
        if mode in ("cross", "mixed"):
            self.build_crossings()
        if mode in ("queue", "mixed") and self.env.length > 70:
            for i, gap in enumerate(NPC_LEAD_GAPS_M):
                s = gap + junction.DefaultVehicle.DEFAULT_LENGTH
                self.spawn(self.env.plan, s, float(self.rng.uniform(*NPC_CRUISE_MPS)), "lead", stopping=i == 1)

    def build_crossings(self):
        plan = self.env.plan
        seen = set()
        for edge_id in plan.edge_ids[:-1]:
            edge = self.net.getEdge(edge_id)
            node = edge.getToNode()
            if node.getID() in seen:
                continue
            seen.add(node.getID())
            candidates = []
            for incoming in node.getIncoming():
                if incoming == edge:
                    continue
                for lane in incoming.getLanes():
                    for connection in lane.getOutgoing():
                        if connection.getDirection() != "s" or not lane.allows("passenger"):
                            continue
                        try:
                            cross_plan = route.plan_route(self.net, lane.getID(), connection.getToLane().getID())
                        except ValueError:
                            continue
                        for ego_s, npc_s in path_crossings(plan, cross_plan):
                            if ego_s > 25 and 10 < npc_s < cross_plan.length_m - 10:
                                candidates.append((ego_s, cross_plan, npc_s, node.getID()))
            if candidates:
                candidate = candidates[int(self.rng.integers(len(candidates)))]
                ego_s, cross_plan, npc_s, node_id = candidate
                self.crossings.append(dict(ego_s=ego_s, plan=cross_plan, npc_s=npc_s, junction_id=node_id,
                                           spawned=0, next_release=0.0))
        self.crossings.sort(key=lambda item: item["ego_s"])

    def spawn(self, plan, station, speed, kind, crossing=None, stopping=False):
        point = self.geometry(plan).extended_position(station)
        vehicles = [self.env.vehicle] + [d.vehicle for d in self.active]
        if any(np.linalg.norm(v.position - point) < 8 for v in vehicles):
            return False
        driver = NpcDriver(self, plan, station, speed, kind, crossing=crossing, stopping=stopping)
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
            if 8 < ahead < CROSS_ACTIVATION_M and item["spawned"] < 2 and self.time >= item["next_release"] and len(self.active) < NPC_MAX_ACTIVE:
                speed = float(self.rng.uniform(*NPC_CRUISE_MPS))
                arrival = ahead / max(self.env.vehicle.speed, 2.5) - float(self.rng.uniform(0.5, 2.0))
                approach = min(item["npc_s"] - 4, float(np.clip(arrival * speed, *CROSS_SPAWN_DISTANCE_M)))
                if self.spawn(item["plan"], item["npc_s"] - approach, speed, "cross",
                              crossing=(item["ego_s"], item["npc_s"], item["junction_id"])):
                    item["spawned"] += 1
                    item["next_release"] = self.time + float(self.rng.uniform(*CROSS_RELEASE_GAP_S))
        for driver in self.active:
            driver.vehicle.before_step(driver.command())

    def after_step(self):
        for driver in self.active:
            was_crossed = driver.crossed
            driver.after_step()
            self.metrics["cross_passages"] += int(driver.crossed and not was_crossed)
            for key, field in (("npc_collision", "collision"), ("npc_red_violation", "red_violation"),
                               ("npc_yellow_violation", "yellow_violation"), ("npc_out_of_lane", "out_of_lane")):
                self.metrics[key] |= bool(getattr(driver, field))
        for i, a in enumerate(self.active):
            for b in self.active[i + 1:]:
                delta = np.asarray(a.vehicle.position - b.vehicle.position)
                if delta @ delta <= (a.radius + b.radius)**2 and traffic.rectangle_clearance(a.vehicle, b.vehicle) <= 0:
                    self.metrics["npc_collision"] = True
        for driver in self.active[:]:
            if driver.finished_route():
                self.remove_events.append(dict(uid=driver.uid, kind=driver.kind, time=self.time, station=driver.station))
                driver.close()
                self.active.remove(driver)

    def close(self):
        for driver in self.active:
            driver.close()
        self.active = []


CONFLICT_SENSOR_M = 65.0
CONFLICT_SIDE_BUFFER_M = 1.3
CONFLICT_TIME_MARGIN_S = 1.8
CONFLICT_STOP_BUFFER_M = 2.5
CONFLICT_BRAKING_MPS2 = 1.8
CONFLICT_MIN_ESTIMATE_MPS = 1.0
CONFLICT_HOLD_THROTTLE = -0.4
CONFLICT_TIME_SCALE_S = 10.0
MIN_BODY_CLEARANCE_M = 0.0


class MultiTrafficObservation(route.DriveObservation):
    @property
    def observation_space(self):
        return junction.gym.spaces.Box(-1, 1, (24,), np.float32)

    def observe(self, vehicle):
        env = self.env
        lead = env.nearest_lead()
        following = np.zeros(4, dtype=np.float32)
        if lead is not None:
            gap, speed, ego = env.lead_state(lead)
            following[:] = (1, gap / traffic.OBS_GAP_M, speed / traffic.OBS_SPEED_MPS,
                            (ego - speed) / traffic.OBS_SPEED_MPS)
        state = env.conflict_state
        cross = np.zeros(6, dtype=np.float32)
        if state:
            cross[:] = (1, state["distance"] / 30, state["arrival_eta"] / CONFLICT_TIME_SCALE_S,
                        state["clear_eta"] / CONFLICT_TIME_SCALE_S,
                        state["ego_clear_eta"] / CONFLICT_TIME_SCALE_S, float(state["blocked"]))
        return np.clip(np.r_[env.drive_obs, following, cross], -1, 1).astype(np.float32)


class MultiTrafficEnv(route.RouteEnv):
    """Follow a planned route among physical queues and crossing traffic."""

    def __init__(self, map_path, *, render=False, traffic="mixed"):
        if traffic not in ("mixed", "queue", "cross", "none"):
            raise ValueError(f"Unknown traffic mode: {traffic}")
        self.traffic = traffic
        self.population = None
        self.conflict_state = None
        self.net = route.read_network(map_path)
        super().__init__(map_path, render=render)

    def get_single_observation(self):
        return MultiTrafficObservation(self.config)

    def reset(self, seed=None, options=None):
        if self.population is not None:
            self.population.close()
        seed = int(self.seed_rng.integers(1, 2**30)) if seed is None else int(seed)
        self.traffic_seed = seed
        self.elapsed = self.headway_duration = self.stop_duration = 0.0
        self.min_clearance_m = self.min_gap_m = self.min_headway_s = math.inf
        self.following_violation = self.headway_violation = self.yield_violation = False
        self.stopped_behind = self.restarted_behind = False
        self.waited_for_cross = self.resumed_after_cross = False
        self.cross_wait_steps = 0
        self.conflict_records = {}
        self.conflict_state = None
        return super().reset(seed=seed, options=options)

    def reset_lights(self, rng):
        super().reset_lights(rng)
        if self.traffic != "none" and not self.sumo_running and self.net.getTrafficLights():
            if junction.libsumo.isLoaded():
                raise RuntimeError("Close the other SUMO simulation before opening this map.")
            junction.libsumo.start(["sumo", "--net-file", self.map_path, "--step-length", str(route.DRIVE_DT_S),
                                    "--no-step-log", "true", "--duration-log.disable", "true"])
            self.sumo_running = True

    def _get_reset_return(self, info):
        if self.population is None:
            self.population = TrafficPopulation(self, self.net)
        self.population.reset(self.traffic_seed, self.traffic)
        result = super()._get_reset_return(info)
        self.update_conflicts()
        observation = self.observations[next(iter(self.agents))].observe(self.vehicle)
        return observation, result[1]

    def nearest_lead(self):
        if self.population is None:
            return None
        candidates = [driver for driver in self.population.active
                      if driver.kind == "lead" and driver.station > self.station]
        return min(candidates, key=lambda driver: driver.station, default=None)

    def lead_state(self, driver):
        gap = driver.station - self.station - (driver.vehicle.LENGTH + self.vehicle.LENGTH) / 2
        speed = max(0., float(self.vehicle.velocity @ self.directions[self.segment_idx]))
        return gap, driver.speed, speed

    def crossing_state(self, driver):
        half = self.vehicle.LENGTH / 2 + driver.vehicle.WIDTH / 2 + CONFLICT_SIDE_BUFFER_M
        other_half = driver.vehicle.LENGTH / 2 + self.vehicle.WIDTH / 2 + CONFLICT_SIDE_BUFFER_M
        entry, leave = driver.ego_cross_s - half, driver.ego_cross_s + half
        npc_entry, npc_leave = driver.cross_s - other_half, driver.cross_s + other_half
        distance = entry - self.station
        if self.station > leave or driver.station > npc_leave or distance > CONFLICT_SENSOR_M:
            return None
        arrival_eta = max(0., npc_entry - driver.station) / max(driver.speed, 0.1)
        clear_eta = max(0., npc_leave - driver.station) / max(driver.speed, 0.1)
        ego_speed = max(CONFLICT_MIN_ESTIMATE_MPS, self.vehicle.speed)
        ego_eta = max(0., distance) / ego_speed
        ego_clear = max(0., leave - self.station) / ego_speed
        lead = self.nearest_lead()
        blocked = (lead is not None and lead.speed < 0.5 and
                   entry < lead.station < leave + self.vehicle.LENGTH + traffic.FOLLOW_STANDSTILL_M)
        overlap = arrival_eta < ego_clear + CONFLICT_TIME_MARGIN_S and clear_eta + CONFLICT_TIME_MARGIN_S > ego_eta
        return dict(driver_id=driver.vehicle.id, ego_entry_s=entry, ego_exit_s=leave,
                    npc_entry_s=npc_entry, npc_exit_s=npc_leave, distance=distance,
                    arrival_eta=arrival_eta, clear_eta=clear_eta, ego_clear_eta=ego_clear,
                    blocked=bool(blocked), unsafe=bool(overlap or blocked),
                    occupied=bool(npc_entry <= driver.station <= npc_leave))

    def update_conflicts(self):
        states = []
        for driver in self.population.active:
            if driver.kind != "cross":
                continue
            state = self.crossing_state(driver)
            if state:
                states.append(state)
        self.conflict_state = None
        if states:
            nearest = min(state["distance"] for state in states)
            nearby = [state for state in states if state["distance"] <= nearest + 4.0]
            threatening = [state for state in nearby if state["unsafe"]]
            candidates = threatening or [min(nearby, key=lambda state: state["clear_eta"])]
            chosen = min(candidates, key=lambda state: state["distance"]).copy()
            chosen["distance"] = nearest
            chosen["ego_entry_s"] = self.station + nearest
            chosen["arrival_eta"] = min(state["arrival_eta"] for state in candidates)
            chosen["clear_eta"] = max(state["clear_eta"] for state in candidates)
            chosen["ego_clear_eta"] = max(state["ego_clear_eta"] for state in candidates)
            chosen["blocked"] = any(state["blocked"] for state in candidates)
            self.conflict_state = chosen

    def teacher_action(self):
        action = super().teacher_action()
        lead = self.nearest_lead()
        if lead is not None:
            gap, speed, ego = self.lead_state(lead)
            target = speed + traffic.FOLLOW_GAP_GAIN * (gap - traffic.FOLLOW_STANDSTILL_M - traffic.FOLLOW_HEADWAY_S * ego)
            if speed < traffic.FOLLOW_STOP_SPEED_MPS and gap <= traffic.FOLLOW_STOP_ZONE_M - 0.4:
                target = 0.
            target = max(0., min(target, self.target_speed))
            action[1] = min(action[1], float(np.clip(traffic.FOLLOW_SPEED_GAIN * (target - ego), -1, 1)))
        state = self.conflict_state
        if state and state["unsafe"] and state["distance"] > -0.3:
            target = math.sqrt(2 * CONFLICT_BRAKING_MPS2 * max(0., state["distance"] - CONFLICT_STOP_BUFFER_M))
            command = 0.8 * (target - self.vehicle.speed)
            if state["distance"] <= CONFLICT_STOP_BUFFER_M + 0.4:
                command = min(command, CONFLICT_HOLD_THROTTLE)
            action[1] = min(action[1], float(np.clip(command, -1, 1)))
        return action

    def _step_simulator(self, actions):
        previous_conflict = self.conflict_state
        self.population.before_step()
        result = super()._step_simulator(actions)
        if self.sumo_running and not self.signals:
            junction.libsumo.simulationStep()
        self.elapsed += route.DRIVE_DT_S
        self.population.after_step()
        self.update_conflicts()
        self.score_traffic(previous_conflict)
        return result

    def score_traffic(self, previous_conflict):
        info = self.step_info
        ego_radius = math.hypot(self.vehicle.LENGTH, self.vehicle.WIDTH) / 2
        for driver in self.population.active:
            lower_bound = float(np.linalg.norm(self.vehicle.position - driver.vehicle.position)) - ego_radius - driver.radius
            clearance = lower_bound
            if lower_bound < self.min_clearance_m or lower_bound <= MIN_BODY_CLEARANCE_M:
                clearance = traffic.rectangle_clearance(self.vehicle, driver.vehicle)
                self.min_clearance_m = min(self.min_clearance_m, clearance)
            if clearance <= MIN_BODY_CLEARANCE_M or self.vehicle.crash_vehicle:
                info.update(collision=True, failed=True, arrived=False, reward=traffic.FAILURE_REWARD)
            if driver.kind != "cross":
                continue
            record = self.conflict_records.setdefault(driver.vehicle.id, dict(
                ego_cross_s=driver.ego_cross_s, npc_cross_s=driver.cross_s,
                ego_enter=None, ego_clear=None, npc_enter=None, npc_clear=None))
            half = self.vehicle.LENGTH / 2 + driver.vehicle.WIDTH / 2 + CONFLICT_SIDE_BUFFER_M
            other_half = driver.vehicle.LENGTH / 2 + self.vehicle.WIDTH / 2 + CONFLICT_SIDE_BUFFER_M
            record["ego_entry_s"], record["ego_exit_s"] = driver.ego_cross_s - half, driver.ego_cross_s + half
            record["npc_entry_s"], record["npc_exit_s"] = driver.cross_s - other_half, driver.cross_s + other_half
            for name, passed in (("ego_enter", self.station >= driver.ego_cross_s - half),
                                 ("ego_clear", self.station > driver.ego_cross_s + half),
                                 ("npc_enter", driver.station >= driver.cross_s - other_half),
                                 ("npc_clear", driver.station > driver.cross_s + other_half)):
                if passed and record[name] is None:
                    record[name] = self.elapsed
            simultaneous = (driver.ego_cross_s - half <= self.station <= driver.ego_cross_s + half and
                            driver.cross_s - other_half <= driver.station <= driver.cross_s + other_half)
            self.yield_violation |= simultaneous
        for record in self.conflict_records.values():
            for name, boundary in (("ego_enter", record["ego_entry_s"]), ("ego_clear", record["ego_exit_s"])):
                if record[name] is None and self.station > boundary:
                    record[name] = self.elapsed
        lead = self.nearest_lead()
        if lead is not None:
            gap, speed, ego = self.lead_state(lead)
            self.min_gap_m = min(self.min_gap_m, gap)
            if ego >= 1:
                self.min_headway_s = min(self.min_headway_s, gap / ego)
            short = ego >= 1 and gap < traffic.FOLLOW_HEADWAY_S * ego
            self.headway_duration = self.headway_duration + route.DRIVE_DT_S if short else 0.
            self.headway_violation |= self.headway_duration + 1e-6 >= traffic.HEADWAY_GRACE_S
            self.following_violation |= gap < traffic.MIN_FOLLOW_GAP_M
            info["reward"] -= traffic.FOLLOW_REWARD_COST * route.DRIVE_DT_S * max(0, traffic.FOLLOW_HEADWAY_S * ego - gap)
            stopped = (self.vehicle.speed <= traffic.FOLLOW_STOP_SPEED_MPS and speed <= traffic.FOLLOW_STOP_SPEED_MPS
                       and traffic.MIN_FOLLOW_GAP_M <= gap <= traffic.FOLLOW_STOP_ZONE_M)
            self.stop_duration = self.stop_duration + route.DRIVE_DT_S if stopped else 0.
            self.stopped_behind |= self.stop_duration + 1e-6 >= traffic.FOLLOW_DWELL_S
            self.restarted_behind |= self.stopped_behind and min(speed, ego) >= 1
        else:
            self.headway_duration = self.stop_duration = 0.
        waiting = (previous_conflict and previous_conflict["unsafe"] and
                   0 < previous_conflict["distance"] < 15 and self.vehicle.speed < 0.3)
        self.cross_wait_steps += int(bool(waiting))
        self.waited_for_cross |= bool(waiting)
        self.resumed_after_cross |= self.waited_for_cross and self.vehicle.speed >= 1
        if self.following_violation or self.yield_violation:
            info.update(failed=True, arrived=False, reward=traffic.FAILURE_REWARD)
        crossings = sum(record["ego_clear"] is not None and record["npc_clear"] is not None
                        for record in self.conflict_records.values())
        metrics = self.population.metrics
        info.update(
            traffic_case=self.traffic, min_gap_m=self.min_gap_m if math.isfinite(self.min_gap_m) else None,
            min_time_headway_s=self.min_headway_s if math.isfinite(self.min_headway_s) else None,
            min_clearance_m=self.min_clearance_m if math.isfinite(self.min_clearance_m) else None,
            following_violation=self.following_violation, headway_violation=self.headway_violation,
            yield_violation=self.yield_violation, waited_for_cross=self.waited_for_cross,
            resumed_after_cross=self.resumed_after_cross, cross_wait_steps=self.cross_wait_steps,
            cross_encounters=crossings, stopped_behind=self.stopped_behind, restarted_behind=self.restarted_behind,
            **metrics)

    def close(self):
        if self.population is not None:
            self.population.close()
        super().close()


class CityControls(traffic.TrafficControls):
    def __init__(self):
        super().__init__()
        self.cross_pedal = torch.nn.Sequential(
            torch.nn.Linear(len(CROSS_INPUTS), 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))

    def forward(self, obs):
        inherited = super().forward(obs[:, :18])
        following = self.cross_pedal(obs[:, CROSS_INPUTS])
        pedal = torch.where(obs[:, 18:19] > 0.5, torch.minimum(inherited[:, 1:], following), inherited[:, 1:])
        return torch.cat((inherited[:, :1], pedal), dim=1)


class CityPolicy(junction.SACPolicy):
    def make_actor(self, features_extractor=None):
        actor = super().make_actor(features_extractor)
        actor.mu = CityControls().to(self.device)
        actor.requires_grad_(False)
        actor.mu.cross_pedal.requires_grad_(True)
        return actor


def extend_parameter(name, value, shape):
    if value.shape == shape:
        return value.clone()
    if value.ndim != 2 or shape != (value.shape[0], value.shape[1] + 6):
        raise ValueError(f"Incompatible parameter {name}: {value.shape} -> {shape}")
    zeros = value.new_zeros(value.shape[0], 6)
    return torch.cat((value, zeros), 1) if name == "log_std.weight" else torch.cat((value[:, :-2], zeros, value[:, -2:]), 1)


def copy_optimizer(source, target):
    targets = dict(target.named_parameters())
    for name, parameter in source.named_parameters():
        destination = targets[name]
        for key, value in source.optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor):
                value = extend_parameter(name, value, destination.shape) if value.ndim else value.clone()
            target.optimizer.state[destination][key] = value
    for old, new in zip(source.optimizer.param_groups, target.optimizer.param_groups):
        new.update({key: value for key, value in old.items() if key != "params"})


def load_city_model(checkpoint, env):
    model = SAC.load(checkpoint, device="cpu", seed=TRAIN_SEED)
    if model.observation_space.shape == (24,) and isinstance(model.actor.mu, CityControls):
        model.set_env(env)
        return model
    if model.observation_space.shape != (18,) or not isinstance(model.actor.mu, traffic.TrafficControls):
        raise ValueError("Use an 18-input route-traffic model or a 24-input city-traffic model.")
    previous = model.policy
    entropy = model.log_ent_coef.detach().clone()
    entropy_optimizer = model.ent_coef_optimizer.state_dict()
    model.observation_space = env.observation_space
    model.set_env(env)
    model.policy_class = CityPolicy
    model.replay_buffer = None
    model._setup_model()
    state = model.actor.state_dict()
    for name, value in previous.actor.state_dict().items():
        state[name] = extend_parameter(name, value, state[name].shape)
    model.actor.load_state_dict(state)
    for source, target in ((previous.critic, model.critic), (previous.critic_target, model.critic_target)):
        sizes = target.state_dict()
        target.load_state_dict({name: extend_parameter(name, value, sizes[name].shape)
                                for name, value in source.state_dict().items()})
    copy_optimizer(previous.actor, model.actor)
    copy_optimizer(previous.critic, model.critic)
    with torch.no_grad():
        model.log_ent_coef.copy_(entropy)
    model.ent_coef_optimizer.load_state_dict(entropy_optimizer)
    return model


def collect_demonstrations(env, model, rng, count, corrections=False):
    states, labels = [], []
    done = True
    for _ in range(count):
        if done:
            obs, _ = env.reset(seed=int(rng.integers(*TRAIN_SEEDS)))
        target = env.teacher_action()
        if obs[18] > 0.5:
            states.append(obs.copy())
            labels.append(target[1])
        action, _ = model.predict(obs, deterministic=True)
        if not corrections:
            action[1] = np.clip(target[1] + rng.normal(0, ACTION_NOISE), -1, 1)
        next_obs, reward, ended, truncated, info = env.step(action)
        done = ended or truncated
        info["TimeLimit.truncated"] = truncated and not ended
        model.replay_buffer.add(obs[None], next_obs[None], action[None], np.array([reward]),
                                np.array([done]), [info])
        obs = next_obs
    if not states:
        raise ValueError("Demonstrations did not encounter crossing traffic.")
    return np.stack(states), np.asarray(labels, dtype=np.float32)


def make_loss(actor, states, labels, rng):
    groups = [np.arange(len(labels))]
    for mask in (labels < -0.3, labels > 0.25, states[:, 2] < 0.03):
        if mask.any():
            groups.append(np.flatnonzero(mask))
    states, labels = torch.from_numpy(states), torch.from_numpy(labels)

    def loss():
        per_group = BATCH_SIZE // (2 * len(groups))
        indices = np.concatenate([rng.choice(g, per_group) for g in groups[1:]] +
                                 [rng.choice(groups[0], BATCH_SIZE - per_group * (len(groups) - 1))])
        predicted = actor.mu.cross_pedal(states[indices][:, CROSS_INPUTS]).tanh().flatten()
        return torch.nn.functional.mse_loss(predicted, labels[indices])

    return loss


def train_city(env, checkpoint, steps):
    model = load_city_model(checkpoint, env)
    model.actor.requires_grad_(False)
    model.actor.mu.cross_pedal.requires_grad_(True)
    rng = np.random.default_rng(TRAIN_SEED)
    states, labels = collect_demonstrations(env, model, rng, DEMO_STEPS)
    clone = torch.optim.Adam(model.actor.mu.cross_pedal.parameters(), lr=CLONE_LR)
    for i, updates in enumerate((CLONE_UPDATES, CORRECTION_UPDATES)):
        if i:
            extra, targets = collect_demonstrations(env, model, rng, CORRECTION_STEPS, corrections=True)
            states, labels = np.vstack((states, extra)), np.concatenate((labels, targets))
        loss = make_loss(model.actor, states, labels, rng)
        for _ in range(updates):
            clone.zero_grad()
            loss().backward()
            clone.step()
        print(f"Imitation round {i + 1}: {len(states)} crossing decisions, {updates} updates.", flush=True)

    def anchor(optimizer, args, kwargs):
        for group in optimizer.param_groups:
            group["lr"] = ACTOR_LR
        with torch.enable_grad():
            (ANCHOR_WEIGHT * loss()).backward()

    old_steps, old_updates = model.num_timesteps, model._n_updates
    model.get_env().seed(int(rng.integers(*TRAIN_SEEDS)))
    hook = model.actor.optimizer.register_step_pre_hook(anchor)
    try:
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
    finally:
        hook.remove()
    print(f"SAC online steps {old_steps} -> {model.num_timesteps}, updates {old_updates} -> {model._n_updates}.", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_FILE)
    parser.add_argument("--train-from", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=ONLINE_STEPS)
    parser.add_argument("--traffic", choices=("mixed", "queue", "cross", "none"), default="mixed")
    parser.add_argument("--pick", action="store_true")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--from-xy", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--to-xy", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.map = args.map or (route.TRAIN_MAP if args.train_from else route.EXAMPLE_MAP)
    explicit = any(x is not None for x in (args.start, args.end, args.from_xy, args.to_xy))
    if bool(args.train_from) != bool(args.output):
        parser.error("Use --train-from and --output together.")
    if args.steps < 1 or not 0 <= args.seed < 2**31 - 1:
        parser.error("Choose positive training steps and a seed from 0 to 2147483646.")
    if args.train_from and (explicit or args.pick or args.render or args.traffic == "none"):
        parser.error("Training automatically selects routes and requires traffic.")
    if args.pick and explicit or args.start and args.from_xy or args.end and args.to_xy:
        parser.error("Choose endpoints by mouse, IDs or coordinates.")
    if not args.map.is_file():
        parser.error(f"Missing map file: {args.map}")
    torch.set_num_threads(1)
    net = route.read_network(args.map)
    if args.train_from:
        env = MultiTrafficEnv(args.map, traffic=args.traffic)
        env.set_routes(route.training_routes(net))
        try:
            model = train_city(env, args.train_from, args.steps)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.output)
            print(f"Saved {args.output}", flush=True)
        finally:
            env.close()
        return 0
    if args.pick or not explicit:
        plan = route.select_on_map(net)
        if plan is None:
            return 0
    else:
        def endpoint(name, xy):
            if name is not None:
                return name, None
            if xy is None:
                parser.error("Supply both endpoints or use --pick.")
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
    env = MultiTrafficEnv(args.map, traffic=args.traffic, render=args.render)
    env.set_route(plan)
    try:
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, ended, truncated, info = env.step(action)
            if args.render:
                env.render(text={"Remaining": f"{info['remaining_m']:.0f} m",
                                 "Speed": f"{env.vehicle.speed_km_h:.1f} km/h"})
            if ended or truncated:
                break
        invalid = any(info.get(k, False) for k in ("following_violation", "headway_violation", "yield_violation",
                                                  "npc_collision", "npc_red_violation", "npc_yellow_violation",
                                                  "npc_out_of_lane"))
        success = bool(info["arrived"] and not truncated and not info.get("failed", False) and not invalid)
        print("Arrived" if success else "Trip failed", flush=True)
        for name in ("trip_m", "endpoint_distance_m", "terminal_speed_mps", "collision",
                     "yield_violation", "out_of_lane", "red_violation", "signals_crossed",
                     "min_clearance_m", "peak_active", "cross_passages", "cross_wait_steps"):
            if name in info:
                print(f"{name}: {info[name]}")
        return 0 if success else 1
    finally:
        env.close()


if __name__ == "__main__":
    from train_sac_city_traffic import main as run
    raise SystemExit(run())

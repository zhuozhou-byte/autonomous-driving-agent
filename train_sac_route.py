"""Plan a legal lane route on a SUMO network and drive it with a MetaDrive SAC policy."""

import argparse
import math
from pathlib import Path

import train_sac_signals as junction
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import dist, isfinite
from xml.etree.ElementTree import iterparse

import numpy as np
import sumolib
from sumolib import geomhelper
from sumolib.miscutils import openz

torch, SAC = junction.torch, junction.SAC
PROJECT_DIR = Path(__file__).resolve().parent
EXAMPLE_MAP = PROJECT_DIR / "sumo_region.net.xml"
TRAIN_MAP = PROJECT_DIR / "sumo_city.net.xml"
PUBLISHED_POLICY = PROJECT_DIR / "models" / "sac_route.zip"
TRAIN_SEED = 217
ROUTE_BANK_SIZE = 60
TRAIN_HOPS = (1, 6)
TEACHER_STEPS = 26_000
CORRECTION_STEPS = 8_000
CLONE_UPDATES = 3_600
CORRECTION_UPDATES = 1_400
ONLINE_STEPS = 16_000
HIDDEN_UNITS = (128, 128)
TRAIN_BATCH = 128
CLONE_LR = 0.001
SAC_LR = 0.0001
SAC_ACTOR_LR = 0.000035
ANCHOR_WEIGHT = 40.0
EXPLORATION_LOG_STD = -3.2
DEMO_NOISE_STD = (0.065, 0.10)
NOISE_HOLD_STEPS = 5


CHANGE_RUN_M = 27.0
CHANGE_MARGIN_M = 4.0
CHANGE_SAMPLE_M = 1.5


DRIVE_DT_S = 0.2
PREVIEW_M = (3.5, 7.0, 13.0)
MAX_CRUISE_MPS = 7.0
CORNER_SPEED_MPS = 2.7
STOP_BUFFER_M = 2.3
BRAKING_MPS2 = 2.0
FINISH_M = 1.25
FINISH_SPEED_MPS = 0.12
FINISH_DWELL_S = 0.6
GOAL_APPROACH_GAIN = 0.7
TIGHT_BEND_CURVATURE = 0.27
NARROW_LANE_M = 3.3
NARROW_BEND_CURVATURE = 0.12
BEND_PREVIEW_STATIONS_M = (-16., -7., -2., 1., 4., 8., 16.)
BEND_OUTWARD_OFFSETS_M = (0., 0.14, 0.73, 1.00, -0.25, -0.15, 0.)
NARROW_BEND_OFFSETS_M = (0., 0.27, 0.50, 0.58, 0.38, -0.03, 0.)
MAX_SPAWN_LATERAL_M = 0.12
SPAWN_SPEED_MPS = (0.0, 2.2)
ROUTE_OBS_SIZE = 14


@dataclass
class RoutePlan:
    lane_ids: tuple
    edge_ids: tuple
    points: np.ndarray
    segment_speeds: np.ndarray
    segment_widths: np.ndarray
    length_m: float
    signals: tuple
    lane_changes: tuple = ()


def read_network(path):
    net = sumolib.net.readNet(str(path), withInternal=True, withPrograms=True)
    net.route_map_path = str(path)
    return net


def list_endpoints(net):
    return sorted(lane.getID() for edge in net.getEdges(withInternal=False)
                  for lane in edge.getLanes() if lane.allows("passenger"))


def _endpoint_lanes(net, name):
    try:
        lane = net.getLane(name)
        if lane.getID() == name and not lane.getEdge().getFunction() and lane.allows("passenger"):
            return [lane]
    except (ValueError, KeyError, IndexError):
        pass
    if net.hasEdge(name) and not net.getEdge(name).getFunction():
        lanes = [lane for lane in net.getEdge(name).getLanes() if lane.allows("passenger")]
        if lanes:
            return lanes
    raise ValueError(f"{name!r} is not an external passenger lane or edge.")


def snap_point(net, x, y):
    """Snap SUMO map coordinates to an external passenger lane. Offset is geometric arc length."""
    if not isfinite(x) or not isfinite(y):
        raise ValueError("Map coordinates must be finite.")
    closest = None
    for edge in net.getEdges(withInternal=False):
        for lane in edge.getLanes():
            if lane.allows("passenger"):
                offset, distance = geomhelper.polygonOffsetAndDistanceToPoint((x, y), lane.getShape())
                candidate = (distance, lane.getID(), offset)
                if closest is None or candidate < closest:
                    closest = candidate
    if closest is None:
        raise ValueError("The map has no external passenger lanes.")
    _, lane_id, offset = closest
    snapped = geomhelper.positionAtShapeOffset(net.getLane(lane_id).getShape(), offset)
    return lane_id, float(offset), tuple(snapped)


def _turn_paths(net, connection):
    """Resolve split junction lanes while retaining the turning movement's destination."""
    destination = connection.getToLane()
    pending = [(connection, ())]
    while pending:
        link, path = pending.pop()
        if not link.allows("passenger") or not destination.allows("passenger"):
            continue
        via_id = link.getViaLaneID()
        if not via_id:
            separation = dist(link.getFromLane().getShape()[-1], destination.getShape()[0])
            internal_exit = bool(link.getFromLane().getEdge().getFunction())
            if link.getDirection() == "s" or (internal_exit and separation < 0.25):
                yield path + ((link, destination),)
            continue
        via = net.getLane(via_id)
        if via.allows("passenger") and all(lane != via for _, lane in path):
            for continuation in via.getOutgoing():
                if continuation.getToLane() == destination:
                    pending.append((continuation, path + ((link, via),)))


def _slice_shape(shape, low, high):
    arc = 0.0
    clipped = [geomhelper.positionAtShapeOffset(shape, low)]
    for a, b in zip(shape, shape[1:]):
        arc += dist(a, b)
        if low < arc < high:
            clipped.append(b)
    clipped.append(geomhelper.positionAtShapeOffset(shape, high))
    return clipped


def _assemble_pieces(pieces):
    points, speeds, widths, signals = [], [], [], []
    lane_ids, edge_ids, changes = [], [], []
    station = 0.0
    for lane, shape, crossing, change_from in pieces:
        if not lane_ids or lane_ids[-1] != lane.getID():
            lane_ids.append(lane.getID())
        edge = lane.getEdge()
        if not edge.getFunction() and (not edge_ids or edge_ids[-1] != edge.getID()):
            edge_ids.append(edge.getID())
        start = station
        speed = min(lane.getSpeed(), change_from.getSpeed()) if change_from else lane.getSpeed()
        width = min(lane.getWidth(), change_from.getWidth()) if change_from else lane.getWidth()
        for point in shape:
            if points:
                segment_length = dist(points[-1], point)
                if segment_length < 1e-6:
                    continue
                station += segment_length
                speeds.append(speed)
                widths.append(width)
            points.append(point)
        if crossing and crossing.getTLSID():
            signals.append((station, crossing.getTLSID(), crossing.getTLLinkIndex()))
        if change_from:
            changes.append((start, station, change_from.getID(), lane.getID()))
    if len(points) < 2:
        raise ValueError("The selected start and destination are the same point.")
    return RoutePlan(tuple(lane_ids), tuple(edge_ids), np.asarray(points), np.asarray(speeds),
                     np.asarray(widths), station, tuple(signals), tuple(changes))


def _make_route(start_lane, turns, start_offset, end_offset):
    lanes = [start_lane, *(lane for _, lane in turns)]
    pieces = []
    for i, lane in enumerate(lanes):
        shape = lane.getShape()
        low = start_offset if i == 0 else 0.0
        high = end_offset if i == len(lanes) - 1 else geomhelper.polyLength(shape)
        pieces.append((lane, _slice_shape(shape, low, high), turns[i][0] if i < len(turns) else None, None))
    return _assemble_pieces(pieces)


def _change_restrictions(net):
    if not hasattr(net, "route_change_restrictions"):
        restrictions = {}
        with openz(net.route_map_path, "rb") as stream:
            for _, lane in iterparse(stream):
                if lane.tag == "lane":
                    for sign, key in ((1, "changeLeft"), (-1, "changeRight")):
                        if key in lane.attrib:
                            classes = lane.get(key).split()
                            restrictions[lane.get("id"), sign] = "passenger" in classes or "all" in classes
                lane.clear()
        net.route_change_restrictions = restrictions
    return net.route_change_restrictions


def _edge_maneuver(entry, exit_lane, low, high, length, permissions):
    if entry == exit_lane:
        if low <= high:
            return [(entry, _slice_shape(entry.getShape(), low, high), None, None)]
        return None
    a, b = entry.getIndex(), exit_lane.getIndex()
    sign = 1 if b > a else -1
    lanes = [entry.getEdge().getLane(i) for i in range(a, b + sign, sign)]
    if any(not lane.allows("passenger") for lane in lanes):
        return None
    if any(not permissions.get((lane.getID(), sign), True) for lane in lanes[:-1]):
        return None
    min_length = min(length(lane) for lane in lanes)
    first, last = low / length(entry), high / length(exit_lane)
    run = CHANGE_RUN_M / min_length
    room = last - first - run * (len(lanes) - 1)
    if room * min_length < 2 * CHANGE_MARGIN_M:
        return None
    f = first + room / 2
    pieces = [(entry, _slice_shape(entry.getShape(), low, f * length(entry)), None, None)]
    for lane, neighbor in zip(lanes, lanes[1:]):
        points = []
        for t in np.linspace(0, 1, int(np.ceil(CHANGE_RUN_M / CHANGE_SAMPLE_M)) + 1):
            u = f + t * run
            p = np.asarray(geomhelper.positionAtShapeOffset(lane.getShape(), u * length(lane)))
            q = np.asarray(geomhelper.positionAtShapeOffset(neighbor.getShape(), u * length(neighbor)))
            if dist(p, q) > 0.6 * (lane.getWidth() + neighbor.getWidth()):
                return None
            blend = t**3 * (10 - 15 * t + 6 * t**2)
            points.append(p + blend * (q - p))
        pieces.append((neighbor, points, None, lane))
        f += run
    pieces.append((exit_lane, _slice_shape(exit_lane.getShape(), f * length(exit_lane), high), None, None))
    return pieces


def _route_with_changes(net, origins, targets, length):
    permissions = _change_restrictions(net)
    queue, previous, distances = [], {}, {}
    ticket = count()
    for lane in origins:
        key = (lane, False)
        distances[key] = 0.0
        heappush(queue, (0.0, next(ticket), key))
    best, last_step = float("inf"), None
    while queue:
        cost, _, key = heappop(queue)
        if cost >= best:
            break
        if cost != distances[key]:
            continue
        entry, moved = key
        low = 0.0 if moved else origins[entry]
        for exit_lane in entry.getEdge().getLanes():
            if exit_lane in targets:
                finish = _edge_maneuver(entry, exit_lane, low, targets[exit_lane], length, permissions)
                if finish:
                    total = cost + sum(geomhelper.polyLength(shape) for _, shape, _, _ in finish)
                    if total < best:
                        best, last_step = total, (key, finish)
            pieces = _edge_maneuver(entry, exit_lane, low, length(exit_lane), length, permissions)
            if not pieces:
                continue
            road_cost = sum(geomhelper.polyLength(shape) for _, shape, _, _ in pieces)
            for connection in exit_lane.getOutgoing():
                for turn in _turn_paths(net, connection):
                    target = (turn[-1][1], True)
                    tail = exit_lane.getShape()[-1]
                    combined = pieces[:-1] + [(exit_lane, pieces[-1][1], connection, None)]
                    next_cost = cost + road_cost
                    for i, (_, lane) in enumerate(turn):
                        next_cost += dist(tail, lane.getShape()[0])
                        if i + 1 < len(turn):
                            combined.append((lane, lane.getShape(), turn[i + 1][0], None))
                            next_cost += length(lane)
                        tail = lane.getShape()[-1]
                    if next_cost < distances.get(target, float("inf")):
                        distances[target] = next_cost
                        previous[target] = key, combined
                        heappush(queue, (next_cost, next(ticket), target))
    if last_step is not None:
        key, finish = last_step
        sections = [finish]
        while key in previous:
            key, pieces = previous[key]
            sections.append(pieces)
        return _assemble_pieces(piece for section in reversed(sections) for piece in section)


def plan_route(net, start, end, *, start_offset=None, end_offset=None):
    """Shortest passenger path along lane connections, including internal junction geometry."""
    starts = _endpoint_lanes(net, start)
    ends = _endpoint_lanes(net, end)
    lengths = {}

    def length(lane):
        if lane not in lengths:
            lengths[lane] = geomhelper.polyLength(lane.getShape())
        return lengths[lane]

    def endpoints(lanes, offset, at_end):
        if offset is not None and (not isfinite(offset) or offset < 0):
            raise ValueError("Lane offsets must be finite, nonnegative metres.")
        positions = {lane: offset if offset is not None else (length(lane) if at_end else 0.0)
                     for lane in lanes if offset is None or offset <= length(lane) + 1e-6}
        if not positions:
            raise ValueError("The selected offset lies beyond the lane endpoint.")
        return {lane: min(pos, length(lane)) for lane, pos in positions.items()}

    origins = endpoints(starts, start_offset, False)
    targets = endpoints(ends, end_offset, True)
    queue, distances, previous = [], {}, {}
    ticket = count()
    for lane, offset in origins.items():
        key = (lane, False)
        distances[key] = -offset
        heappush(queue, (-offset, next(ticket), key))
    best, winner = float("inf"), None
    while queue:
        cost, _, key = heappop(queue)
        if cost >= best:
            break
        if cost != distances[key]:
            continue
        lane, traversed = key
        if lane in targets and (traversed or targets[lane] >= origins[lane]):
            finish_cost = cost + targets[lane]
            if finish_cost < best:
                best, winner = finish_cost, key
        for connection in lane.getOutgoing():
            for turn in _turn_paths(net, connection):
                target = (turn[-1][1], True)
                next_cost = cost + length(lane)
                tail = lane.getShape()[-1]
                for _, entered in turn:
                    next_cost += dist(tail, entered.getShape()[0])
                    if entered != target[0]:
                        next_cost += length(entered)
                    tail = entered.getShape()[-1]
                if next_cost < distances.get(target, float("inf")):
                    distances[target] = next_cost
                    previous[target] = (key, turn)
                    heappush(queue, (next_cost, next(ticket), target))
    if winner is None:
        if hasattr(net, "route_map_path") and any(len(e.getLanes()) > 1 for e in net.getEdges(withInternal=False)):
            plan = _route_with_changes(net, origins, targets, length)
            if plan is not None:
                return plan
        raise ValueError(f"No passenger route connects {start!r} to {end!r} by legal lane connections.")
    key = winner
    sections = []
    while key in previous:
        key, turn = previous[key]
        sections.append(turn)
    turns = [step for section in reversed(sections) for step in section]
    return _make_route(key[0], turns, origins[key[0]], targets[winner[0]])


class DriveObservation(junction.BaseObservation):
    @property
    def observation_space(self):
        return junction.gym.spaces.Box(-1, 1, (ROUTE_OBS_SIZE,), np.float32)

    def reset(self, env, vehicle=None):
        self.env = env

    def observe(self, vehicle):
        return self.env.drive_obs.copy()


class RouteEnv(junction.BaseEnv):
    """Drive a physical MetaDrive vehicle along a directed SUMO lane itinerary."""

    def __init__(self, map_path, *, render=False):
        self.map_path = str(Path(map_path).resolve())
        self.routes = []
        self.seed_rng = np.random.default_rng(173)
        self.sumo_running = False
        self.drive_obs = np.zeros(ROUTE_OBS_SIZE, dtype=np.float32)
        super().__init__(dict(use_render=render, log_level=junction.logging.ERROR,
                              num_scenarios=2**31 - 1, horizon=1500,
                              physics_world_step_size=0.02, decision_repeat=10,
                              agent_observation=DriveObservation,
                              vehicle_config=dict(navigation_module=None, show_navi_mark=False,
                                                  show_dest_mark=False, spawn_velocity_car_frame=True)))

    def setup_engine(self):
        super().setup_engine()
        self.engine.register_manager("map_manager", junction.SumoMapManager(self.map_path))

    def set_route(self, plan):
        self.set_routes([plan])

    def set_routes(self, plans):
        self.routes = list(plans)
        if not self.routes:
            raise ValueError("Supply at least one reachable route.")

    def reset(self, seed=None, options=None):
        if not self.routes:
            raise ValueError("Call set_route or set_routes before reset.")
        seed = int(self.seed_rng.integers(1, 2**30)) if seed is None else int(seed)
        rng = np.random.default_rng(seed)
        self.plan = self.routes[int(rng.integers(len(self.routes)))]
        self.points = np.asarray(self.plan.points, dtype=float)
        vectors = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(vectors, axis=1)
        self.directions = vectors / self.lengths[:, None]
        self.stations = np.r_[0., self.lengths.cumsum()]
        angles = np.unwrap(np.arctan2(self.directions[:, 1], self.directions[:, 0]))
        curvature = np.diff(angles) / (0.5 * (self.lengths[:-1] + self.lengths[1:]))
        narrow = np.minimum(self.plan.segment_widths[:-1], self.plan.segment_widths[1:]) < NARROW_LANE_M
        sharp = np.flatnonzero(np.abs(curvature) > np.where(narrow, NARROW_BEND_CURVATURE, TIGHT_BEND_CURVATURE))
        separated = ((np.diff(self.stations[sharp + 1]) > 7.) |
                     (np.diff(np.sign(curvature[sharp])) != 0))
        bends = [group for group in np.split(sharp, np.flatnonzero(separated) + 1) if len(group)]
        self.bend_stations = np.array([self.stations[group + 1].mean() for group in bends])
        self.bend_signs = np.array([np.sign(curvature[group[0]]) for group in bends])
        self.bend_offsets = [NARROW_BEND_OFFSETS_M if narrow[group].all() and
                             np.max(abs(curvature[group])) <= TIGHT_BEND_CURVATURE else BEND_OUTWARD_OFFSETS_M
                             for group in bends]
        self.length = float(self.stations[-1])
        self.finish_radius = min(FINISH_M, self.length * 0.15)
        self.signals = self.plan.signals
        self.signal_index = self.signals_crossed = self.red_wait_steps = 0
        self.had_overspeed = False
        self.station = 0.0
        self.goal_dwell_s = 0.0
        self.step_info = {}
        tangent = self.directions[0]
        spawn_jitter = min(MAX_SPAWN_LATERAL_M, self.finish_radius / 3)
        lateral = rng.uniform(-spawn_jitter, spawn_jitter)
        pos = self.points[0] + lateral * np.array([-tangent[1], tangent[0]])
        heading = math.atan2(tangent[1], tangent[0])
        self.lazy_init()
        spawn = self.config["agent_configs"]["default_agent"]
        spawn["spawn_position_heading"] = (tuple(pos), heading)
        spawn["spawn_velocity"] = (min(float(rng.uniform(*SPAWN_SPEED_MPS)), self.length * 0.4), 0.)
        self.config["horizon"] = int((20 + self.length / 1.3 + 120 * len(self.signals)) / DRIVE_DT_S)
        self.reset_lights(rng)
        return super().reset(seed=seed)

    def reset_lights(self, rng):
        if not self.signals:
            return
        sumo = junction.libsumo
        if not self.sumo_running:
            if sumo.isLoaded():
                raise RuntimeError("Close the other SUMO simulation before opening this map.")
            sumo.start(["sumo", "--net-file", self.map_path, "--step-length", str(DRIVE_DT_S),
                        "--no-step-log", "true", "--duration-log.disable", "true"])
            self.sumo_running = True
        lights = sumo.trafficlight
        for tls in sorted({tls for _, tls, _ in self.signals}):
            program = lights.getProgram(tls)
            logic = next(p for p in lights.getAllProgramLogics(tls) if p.programID == program)
            durations = np.array([phase.duration for phase in logic.phases])
            offset = float(rng.uniform(0.0, durations.sum()))
            ends = durations.cumsum()
            phase = int(ends.searchsorted(offset, side="right"))
            lights.setPhase(tls, phase)
            lights.setPhaseDuration(tls, float(ends[phase] - offset))

    def segment(self, station):
        return min(max(int(self.stations.searchsorted(station, side="right")) - 1, 0), len(self.lengths) - 1)

    def position_at(self, station):
        s = min(max(0.0, station), self.length)
        i = self.segment(s)
        center = self.points[i] + self.directions[i] * (s - self.stations[i])
        if len(self.bend_stations):
            bend = int(np.argmin(abs(self.bend_stations - s)))
            offset = np.interp(s - self.bend_stations[bend], BEND_PREVIEW_STATIONS_M, self.bend_offsets[bend])
            normal = np.array([-self.directions[i, 1], self.directions[i, 0]])
            center -= offset * self.bend_signs[bend] * normal
        return center

    def project(self, position, near):
        # A local window prevents jumps to later sections at overpasses or close parallel lanes.
        a, b = self.segment(near - 9), self.segment(near + 9) + 1
        delta = np.asarray(position) - self.points[a:b]
        dirs = self.directions[a:b]
        along = np.clip((delta * dirs).sum(axis=1), 0, self.lengths[a:b])
        residual = delta - along[:, None] * dirs
        k = int(np.argmin((residual * residual).sum(axis=1)))
        lat = dirs[k, 0] * residual[k, 1] - dirs[k, 1] * residual[k, 0]
        i = a + k
        return float(self.stations[i] + along[k]), float(lat), i

    def read_light(self):
        if self.signal_index >= len(self.signals):
            return math.inf, "g"
        station, tls, link = self.signals[self.signal_index]
        distance = station - self.station - self.vehicle.LENGTH / 2
        state = junction.libsumo.trafficlight.getRedYellowGreenState(tls)[link].lower()
        return distance, state

    def refresh(self):
        v = self.vehicle
        self.station, self.lateral, i = self.project(v.position, self.station)
        self.segment_idx = i
        heading = math.atan2(self.directions[i, 1], self.directions[i, 0])
        self.heading_error = v.heading_theta - heading
        self.speed_limit = self.plan.segment_speeds[i]
        self.dist_light, self.light_state = self.read_light()
        if self.light_state != "y":
            self.clear_yellow = False
        elif self.previous_light != "y":
            self.clear_yellow = self.dist_light < v.speed * DRIVE_DT_S + v.speed**2 / (2 * 3.5) + 0.8
        self.previous_light = self.light_state
        self.should_stop = self.light_state in ("r", "u") or (self.light_state == "y" and not self.clear_yellow)
        bearings = []
        for ahead in PREVIEW_M:
            delta = self.position_at(self.station + ahead) - v.position
            if self.station + ahead > self.length:
                delta += self.directions[-1] * (self.station + ahead - self.length)
            bearings.append(math.atan2(delta[1], delta[0]) - v.heading_theta)
        steering_preview = math.sin(bearings[0])
        if len(self.bend_stations) and np.min(abs(self.bend_stations - self.station)) < 10:
            steering_preview *= 1.05
        a, b = self.segment(self.station - 5), self.segment(self.station + 12) + 1
        curve = np.max(np.abs(self.directions[a:b, 0] * self.directions[i, 1]
                              - self.directions[a:b, 1] * self.directions[i, 0]))
        target = min(MAX_CRUISE_MPS, 0.86 * min(self.plan.segment_speeds[a:b]),
                     MAX_CRUISE_MPS - (MAX_CRUISE_MPS - CORNER_SPEED_MPS) * min(1., 2.0 * curve))
        remaining = self.length - self.station
        goal_scale = min(1., self.finish_radius / FINISH_M)
        stopping_room = max(0., remaining - 0.48 * self.finish_radius)
        goal_speed = GOAL_APPROACH_GAIN * stopping_room / goal_scale
        if goal_scale < 1:
            goal_speed = min(goal_speed, math.sqrt(2 * BRAKING_MPS2 * stopping_room))
        target = min(target, goal_speed)
        if self.should_stop:
            target = min(target, math.sqrt(2 * BRAKING_MPS2 * max(0., self.dist_light - STOP_BUFFER_M)))
        self.target_speed = target
        self.drive_obs = np.clip(np.array([
            self.lateral / (self.plan.segment_widths[i] / 2), math.sin(self.heading_error), v.speed / 10,
            steering_preview, math.sin(bearings[1]), math.sin(bearings[2]),
            target / 10, (target - v.speed) / 6,
            self.speed_limit / 15, self.dist_light / 35, float(self.light_state in ("r", "u")),
            float(self.light_state == "y"), float(self.clear_yellow), remaining / (25 * goal_scale),
        ], dtype=np.float32), -1, 1)

    def _get_reset_return(self, info):
        self.previous_light = ""
        self.clear_yellow = False
        self.refresh()
        while self.dist_light <= 0 and self.signal_index < len(self.signals):
            self.signal_index += 1
            self.previous_light = ""
            self.refresh()
        return super()._get_reset_return(info)

    def teacher_action(self):
        v = self.vehicle
        wheelbase = v.FRONT_WHEELBASE + v.REAR_WHEELBASE
        steer = math.atan(2 * wheelbase * self.drive_obs[3] / PREVIEW_M[0]) / math.radians(v.max_steering)
        throttle = 0.65 * (self.target_speed - v.speed)
        if self.should_stop and self.dist_light <= STOP_BUFFER_M + 0.3:
            throttle = min(throttle, -0.3)
        if self.length - self.station <= 0.65 * self.finish_radius:
            throttle = min(throttle, -0.3)
        return np.clip([steer, throttle], -1, 1).astype(np.float32)

    def _step_simulator(self, actions):
        old_s, old_d, old_light, could_clear = self.station, self.dist_light, self.light_state, self.clear_yellow
        info = super()._step_simulator(actions)
        v = self.vehicle
        self.station, self.lateral, i = self.project(v.position, old_s)
        crossing = self.signal_index < len(self.signals) and old_d > 0 >= self.read_light()[0]
        red = crossing and old_light in ("r", "u")
        yellow = crossing and old_light == "y" and not could_clear
        if crossing:
            self.signals_crossed += 1
            self.signal_index += 1
            self.previous_light = ""
        if self.signals:
            junction.libsumo.simulationStep()
        self.refresh()
        self.red_wait_steps += int(old_light in ("r", "u") and 0 < old_d < 8 and v.speed < 0.3)
        width = self.plan.segment_widths[i]
        half_extent = (v.WIDTH * abs(math.cos(self.heading_error)) + v.LENGTH * abs(math.sin(self.heading_error))) / 2
        out = abs(self.lateral) + half_extent > width / 2
        if out or self.segment(old_s - 3.0) != self.segment(self.station + 3.0):
            out = any(abs(lateral) > self.plan.segment_widths[segment] / 2
                      for corner in junction.vehicle_corners(v)
                      for _, lateral, segment in [self.project(corner, self.station)])
        wrong = math.cos(self.heading_error) <= 0
        collision = bool(v.crash_vehicle or v.crash_object or v.crash_human or v.crash_sidewalk or v.crash_building)
        failed = bool(out or wrong or collision or red or yellow)
        remaining = self.length - self.station
        endpoint = float(np.linalg.norm(v.position - self.points[-1]))
        passed_goal = (i == len(self.lengths) - 1
                       and float((v.position - self.points[-1]) @ self.directions[-1]) > self.finish_radius)
        failed |= passed_goal
        parked = remaining <= self.finish_radius and endpoint <= self.finish_radius and v.speed <= FINISH_SPEED_MPS
        self.goal_dwell_s = self.goal_dwell_s + DRIVE_DT_S if parked else 0.0
        arrived = bool(self.goal_dwell_s + 1e-6 >= FINISH_DWELL_S and not failed)
        overspeed = max(0.0, v.speed - self.speed_limit)
        self.had_overspeed |= overspeed > 0
        reward = (self.station - old_s) - DRIVE_DT_S * (0.2 * abs(self.lateral) + 2.0 * overspeed)
        reward = -50.0 if failed else reward + 30.0 * arrived
        self.step_info = dict(arrived=arrived, failed=failed, collision=collision, out_of_lane=bool(out),
                              wrong_way=bool(wrong), red_violation=bool(red), yellow_violation=bool(yellow),
                              had_overspeed=bool(self.had_overspeed), trip_m=self.station,
                              route_length_m=self.length, remaining_m=remaining, endpoint_distance_m=endpoint,
                              goal_overshoot=passed_goal, terminal_speed_mps=v.speed, goal_dwell_s=self.goal_dwell_s,
                              signals_crossed=self.signals_crossed, red_wait_steps=self.red_wait_steps, reward=reward)
        return info

    def reward_function(self, vehicle_id):
        return float(self.step_info.get("reward", 0)), self.step_info

    def done_function(self, vehicle_id):
        done = self.step_info.get("failed", False) or self.step_info.get("arrived", False)
        return bool(done), {"max_step": self.episode_lengths[vehicle_id] >= self.config["horizon"] and not done}

    def cost_function(self, vehicle_id):
        return 0.0, {}

    def close(self):
        try:
            if self.sumo_running:
                junction.libsumo.close()
                self.sumo_running = False
        finally:
            super().close()


class RouteControls(torch.nn.Module):
    """Use the route bearing for steering and the speed and stop observations for throttle."""

    def __init__(self):
        super().__init__()
        self.steer = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(),
                                         torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
        self.pedal = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                                         torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))

    def forward(self, obs):
        bearing = obs[:, 3:4]
        steering = (self.steer(bearing) - self.steer(-bearing)) * 0.5
        pedal_inputs = obs[:, [2, 6, 7, 9, 10, 11, 12, 13]]
        return torch.cat((steering, self.pedal(pedal_inputs)), dim=1)


class RoutePolicy(junction.SACPolicy):
    def make_actor(self, features_extractor=None):
        actor = super().make_actor(features_extractor)
        actor.mu = RouteControls().to(self.device)
        return actor


def training_routes(net, seed=TRAIN_SEED):
    rng = np.random.default_rng(seed)
    endpoints = list_endpoints(net)
    if not endpoints:
        raise ValueError("This network has no passenger lanes.")
    plans = {}
    for _ in range(ROUTE_BANK_SIZE * 20):
        start = str(rng.choice(endpoints))
        lane = net.getLane(start)
        for _ in range(int(rng.integers(TRAIN_HOPS[0], TRAIN_HOPS[1] + 1))):
            turns = [c for c in lane.getOutgoing() if c.getToLane().allows("passenger")]
            if not turns:
                break
            forward = [c for c in turns if c.getDirection() != "t"]
            options = forward or turns
            lane = options[int(rng.integers(len(options)))].getToLane()
        end = lane.getID()
        if start == end or (start, end) in plans:
            continue
        try:
            plan = plan_route(net, start, end)
        except ValueError:
            continue
        plans[start, end] = plan
        if len(plans) >= ROUTE_BANK_SIZE:
            break
    if not plans:
        raise ValueError("Training needs at least two connected passenger lanes.")
    return list(plans.values())


def collect_actions(env, model, rng, steps, *, corrections=False):
    states, targets = [], []
    finished = True
    for i in range(steps):
        if finished:
            obs, _ = env.reset(seed=int(rng.integers(100_000, 800_000)))
        target = env.teacher_action()
        states.append(obs.copy())
        targets.append(target.copy())
        if corrections:
            action, _ = model.predict(obs, deterministic=True)
        else:
            if finished or i % NOISE_HOLD_STEPS == 0:
                noise = rng.normal(scale=DEMO_NOISE_STD)
            action = np.clip(target + noise, -1, 1).astype(np.float32)
        next_obs, reward, ended, truncated, info = env.step(action)
        finished = ended or truncated
        info["TimeLimit.truncated"] = truncated and not ended
        model.replay_buffer.add(obs[None], next_obs[None], action[None], np.array([reward]),
                                np.array([finished]), [info])
        obs = next_obs
    return np.stack(states), np.stack(targets)


def imitation_batches(actor, states, targets, rng):
    groups = [np.arange(len(targets))]
    for mask in (abs(targets[:, 0]) > 0.68, abs(targets[:, 0]) > 0.12, states[:, 13] < 0.08,
                 targets[:, 1] < -0.25, targets[:, 1] > 0.35):
        if mask.any():
            groups.append(np.flatnonzero(mask))
    states = torch.from_numpy(states)
    targets = torch.from_numpy(targets)

    def loss():
        n = TRAIN_BATCH // (2 * len(groups))
        indices = np.concatenate([rng.choice(g, n) for g in groups[1:]] +
                                 [rng.choice(groups[0], TRAIN_BATCH - n * (len(groups) - 1))])
        mean = actor.mu(states[indices])
        target = targets[indices]
        logit = torch.atanh(target[:, 0].clamp(-0.99, 0.99))
        return (torch.nn.functional.mse_loss(mean.tanh(), target) +
                0.1 * torch.nn.functional.smooth_l1_loss(mean[:, 0], logit))

    return loss


def train_route_driver(env, net, online_steps):
    plans = training_routes(net)
    env.set_routes(plans)
    model = SAC(RoutePolicy, env, learning_rate=SAC_LR, buffer_size=TEACHER_STEPS + CORRECTION_STEPS + online_steps,
                learning_starts=0, train_freq=4, gradient_steps=1, batch_size=TRAIN_BATCH,
                ent_coef="auto_0.01", policy_kwargs=dict(net_arch=dict(pi=[], qf=list(HIDDEN_UNITS))),
                seed=TRAIN_SEED, device="cpu")
    actor = model.actor
    torch.nn.init.zeros_(actor.log_std.weight)
    torch.nn.init.constant_(actor.log_std.bias, EXPLORATION_LOG_STD)
    rng = np.random.default_rng(TRAIN_SEED)
    states, targets = collect_actions(env, model, rng, TEACHER_STEPS)
    clone = torch.optim.Adam(actor.parameters(), lr=CLONE_LR)
    for round_index, updates in enumerate((CLONE_UPDATES, CORRECTION_UPDATES)):
        if round_index:
            extra_states, extra_targets = collect_actions(env, model, rng, CORRECTION_STEPS, corrections=True)
            states, targets = np.vstack((states, extra_states)), np.vstack((targets, extra_targets))
        loss = imitation_batches(actor, states, targets, rng)
        for _ in range(updates):
            clone.zero_grad()
            loss().backward()
            clone.step()
        print(f"Imitation round {round_index + 1}: {len(states)} decisions, {updates} updates.", flush=True)

    def anchor_driver(optimizer, args, kwargs):
        for group in optimizer.param_groups:
            group["lr"] = SAC_ACTOR_LR
        with torch.enable_grad():
            (ANCHOR_WEIGHT * loss()).backward()

    hook = actor.optimizer.register_step_pre_hook(anchor_driver)
    try:
        model.learn(total_timesteps=online_steps)
    finally:
        hook.remove()
    print(f"Trained on {len(plans)} routes, {model.num_timesteps} online steps, {model._n_updates} SAC updates.")
    return model


def select_on_map(net):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(11, 8))
    lanes = list_endpoints(net)
    if not lanes:
        raise ValueError("The map has no passenger lanes.")
    ax.add_collection(LineCollection([net.getLane(lane).getShape() for lane in lanes],
                                     colors="#81919a", linewidths=1))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set(xlabel="SUMO x (m)", ylabel="SUMO y (m)")
    start_mark, = ax.plot([], [], "o", color="#128247", zorder=4)
    end_mark, = ax.plot([], [], "o", color="#bd482f", zorder=4)
    line, = ax.plot([], [], color="#006ac3", linewidth=2.5)
    start, plan, accepted = None, None, False
    initial = "Zoom to a road, then click the start lane. Backspace resets the selection."
    ax.set_title(initial)

    def click(event):
        nonlocal start, plan
        toolbar = getattr(fig.canvas.manager, "toolbar", None)
        if event.inaxes is not ax or event.button != 1 or getattr(toolbar, "mode", "") or plan is not None:
            return
        lane, offset, xy = snap_point(net, event.xdata, event.ydata)
        if start is None:
            start = lane, offset
            start_mark.set_data([xy[0]], [xy[1]])
            ax.set_title(f"Start: {lane}. Click the destination lane.")
        else:
            try:
                plan = plan_route(net, start[0], lane, start_offset=start[1], end_offset=offset)
            except ValueError as exc:
                ax.set_title(f"{exc}\nChoose a different destination or press Backspace.")
            else:
                end_mark.set_data([xy[0]], [xy[1]])
                line.set_data(plan.points[:, 0], plan.points[:, 1])
                ax.set_title(f"{start[0]} to {lane} | {plan.length_m:.0f} m | Enter to confirm, Backspace to reset")
        fig.canvas.draw_idle()

    def key(event):
        nonlocal start, plan, accepted
        if event.key == "enter" and plan is not None:
            accepted = True
            plt.close(fig)
        elif event.key == "backspace":
            start = plan = None
            for artist in (start_mark, end_mark, line):
                artist.set_data([], [])
            ax.set_title(initial)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", click)
    fig.canvas.mpl_connect("key_press_event", key)
    plt.show(block=True)
    return plan if accepted else None


def drive_route(env, plan, model, seed, render):
    env.set_route(plan)
    obs, _ = env.reset(seed=seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, ended, truncated, info = env.step(action)
        if render:
            status = {"Remaining": f"{info['remaining_m']:.0f} m", "Speed": f"{env.vehicle.speed_km_h:.1f} km/h"}
            if env.signal_index < len(plan.signals):
                color = {"r": "red", "u": "red/yellow", "y": "yellow", "g": "green"}.get(env.light_state, env.light_state)
                status["Next signal"] = f"{color}, {env.dist_light:.0f} m"
            env.render(text=status)
        if ended or truncated:
            break
    failures = [k for k in ("collision", "out_of_lane", "wrong_way", "red_violation", "yellow_violation",
                            "goal_overshoot") if info.get(k)]
    if truncated:
        failures.append("timeout")
    outcome = "Arrived" if info["arrived"] else "Failed (" + ", ".join(failures or ["route incomplete"]) + ")"
    print(f"{outcome}. Travelled {info['trip_m']:.1f}/{plan.length_m:.1f} m; "
          f"distance to destination {info['endpoint_distance_m']:.2f} m; speed {env.vehicle.speed:.2f} m/s; "
          f"passed {info['signals_crossed']} signals.")
    return info["arrived"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, help="A SUMO .net.xml road network.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Train a route driver on this map.")
    mode.add_argument("--list", "--list-lanes", action="store_true", help="List passenger lane IDs.")
    mode.add_argument("--pick", action="store_true", help="Choose two positions on the map with the mouse.")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--start", help="Start edge or lane ID, at its upstream end.")
    start.add_argument("--from-xy", nargs=2, type=float, metavar=("X", "Y"), help="Start in SUMO map metres.")
    end = parser.add_mutually_exclusive_group()
    end.add_argument("--end", help="Destination edge or lane ID, at its downstream end.")
    end.add_argument("--to-xy", nargs=2, type=float, metavar=("X", "Y"), help="Destination in SUMO map metres.")
    parser.add_argument("--plan-only", action="store_true", help="Print the chosen route and exit.")
    parser.add_argument("--render", action="store_true", help="Show the physical drive in MetaDrive.")
    parser.add_argument("--model", type=Path, default=PUBLISHED_POLICY)
    parser.add_argument("--output", type=Path, help="Checkpoint path for --train.")
    parser.add_argument("--steps", type=int, default=ONLINE_STEPS, help="Online SAC steps for training.")
    parser.add_argument("--seed", type=int, default=42, help="Initial signal phases and vehicle state for driving.")
    args = parser.parse_args()
    args.map = args.map or (TRAIN_MAP if args.train else EXAMPLE_MAP)
    if not args.map.is_file():
        parser.error(f"Map file does not exist: {args.map}")
    explicit = any(x is not None for x in (args.start, args.from_xy, args.end, args.to_xy))
    if (args.pick or args.train or args.list) and explicit:
        parser.error("Choose a mode or a pair of endpoints.")
    if bool(args.output) != args.train or args.steps < 1:
        parser.error("Use --output with --train and choose a positive step count.")
    if args.train and (args.plan_only or args.render):
        parser.error("Use --plan-only and --render when selecting a route.")
    if args.seed < 0 or args.seed >= 2**31 - 1:
        parser.error("Seed must be between 0 and 2147483646.")

    torch.set_num_threads(1)
    net = read_network(args.map)
    if args.list:
        print("\n".join(list_endpoints(net)))
        return 0
    if args.train:
        env = RouteEnv(args.map)
        try:
            model = train_route_driver(env, net, args.steps)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.output)
        finally:
            env.close()
        return 0
    if args.pick or not explicit:
        plan = select_on_map(net)
        if plan is None:
            return 0
    else:
        def endpoint(name, xy):
            if name is not None:
                return name, None
            if xy is None:
                parser.error("Supply both endpoints, or use --pick.")
            lane, offset, point = snap_point(net, *xy)
            print(f"Selected {lane} at ({point[0]:.2f}, {point[1]:.2f}), {math.dist(xy, point):.2f} m from input.")
            return lane, offset

        try:
            origin, begin = endpoint(args.start, args.from_xy)
            goal, finish = endpoint(args.end, args.to_xy)
            plan = plan_route(net, origin, goal, start_offset=begin, end_offset=finish)
        except ValueError as exc:
            parser.error(str(exc))
    print("Route: " + " -> ".join(plan.edge_ids))
    print(f"{plan.length_m:.1f} m, {len(plan.signals)} traffic signals.", flush=True)
    if args.plan_only:
        return 0
    if not args.model.is_file():
        parser.error(f"Driver checkpoint does not exist: {args.model}")
    model = SAC.load(args.model, device="cpu")
    env = RouteEnv(args.map, render=args.render)
    try:
        return 0 if drive_route(env, plan, model, args.seed, args.render) else 1
    finally:
        env.close()


if __name__ == "__main__":
    from train_sac_route import main as run
    raise SystemExit(run())

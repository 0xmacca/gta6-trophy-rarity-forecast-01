"""GTA6 trophy rarity forecast entry.

Cohort-survival model: rate = earners / players_tracked.
Two methods (A: single anchor curve; C: 225-scenario weighted grid)
are blended 50/50 in logit space.

Standard library only. Deterministic. No randomness, no wall-clock.
"""

import bisect
import json
import math
import os
from datetime import datetime, timezone

LAUNCH_DT = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LAUNCH_PLAYERS = 638

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

SPREAD_SHORT = 0.35
SPREAD_MULT = 0.25

# Trophy classes by id.
_CLASS_BY_ID = {}
for _i in (1,):
    _CLASS_BY_ID[_i] = "platinum"
for _i in (2,):
    _CLASS_BY_ID[_i] = "fishing"
for _i in (3, 4, 5, 6):
    _CLASS_BY_ID[_i] = "chapter"
for _i in (7,):
    _CLASS_BY_ID[_i] = "final"
for _i in (14,):
    _CLASS_BY_ID[_i] = "comp100"
for _i in (8, 9, 10, 11, 12, 13, 22, 23, 24, 26, 27):
    _CLASS_BY_ID[_i] = "strand"
for _i in (15, 17, 25, 30, 31, 40):
    _CLASS_BY_ID[_i] = "grind_silver"
for _i in (16, 18, 19, 20, 21, 28, 29, 32, 35, 52, 53, 54, 55):
    _CLASS_BY_ID[_i] = "grind_bronze"


def trophy_class(tid):
    return _CLASS_BY_ID.get(tid, "oneshot")


# Shared per-class (A, tau in days) centres.
CENTRES = {
    "fishing": (0.999, 0.01),
    "chapter": (0.40, 0.80),
    "final": (0.25, 1.50),
    "strand": (0.35, 1.00),
    "oneshot": (0.25, 0.80),
    "grind_bronze": (0.03, 3.00),
    "grind_silver": (0.012, 4.00),
    "platinum": (0.006, 4.00),
    "comp100": (0.002, 5.00),
}

# Method A uses the same centres except fishing.
CENTRES_A = dict(CENTRES)
CENTRES_A["fishing"] = (1.0, 0.005)

# Observed 24h rate ratios (rate at 24h / launch rate), by trophy id.
_RATIO_BY_ID = {
    2: 1.00,
    20: 0.93,
    3: 0.58,
    21: 0.44,
    35: 0.31,
    8: 0.22,
    37: 0.15,
    38: 0.15,
    1: 0.05,
    14: 0.05,
}
_CLASS_RATIO_DEFAULT = {"fishing": 1.00}


def obs_ratio(tid):
    if tid in _RATIO_BY_ID:
        return _RATIO_BY_ID[tid]
    return _CLASS_RATIO_DEFAULT.get(trophy_class(tid), 0.05)


# Method A per-class logit sigma.
SIGMA_CLASS = {
    "fishing": 0.35,
    "chapter": 0.45,
    "final": 0.50,
    "strand": 0.48,
    "oneshot": 0.45,
    "grind_bronze": 0.45,
    "grind_silver": 0.45,
    "platinum": 0.35,
    "comp100": 0.40,
}

# Method A short-horizon join fractions.
JOIN_FRAC = {
    "fishing": 1.0,
    "chapter": 0.10,
    "strand": 0.10,
    "oneshot": 0.07,
    "grind_bronze": 0.03,
    "grind_silver": 0.02,
    "platinum": 0.01,
    "comp100": 0.01,
    "final": 0.05,
}

# Method A anchor curve for N(t).
ANCHORS_A = [(0.0, 638.0), (0.1875, 3868.0), (1.0, 95000.0), (7.0, 210000.0)]

# Method C prior grid.
GRID_N24 = [51000.0, 73000.0, 95000.0, 117000.0, 139000.0]
GRID_NFIN = [78000.0, 114000.0, 210000.0, 330000.0, 510000.0]
GRID_A_MULT = [(0.65, 0.25), (1.05, 0.50), (1.45, 0.25)]
GRID_TAU_MULT = [(0.75, 0.25), (1.05, 0.50), (1.35, 0.25)]


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def clamp(x):
    if x < 0.01:
        return 0.01
    if x > 99.99:
        return 99.99
    return x


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(L):
    # Stable percent form of 100 / (1 + exp(-L)).
    if L >= 0:
        return 100.0 / (1.0 + math.exp(-L))
    e = math.exp(L)
    return 100.0 * e / (1.0 + e)


def round4(x):
    v = math.floor(x * 10000.0 + 0.5) / 10000.0
    if v == int(v):
        return int(v)
    return v


def parse_time(value):
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def days_since_launch(dt, release_dt):
    if dt is None or release_dt is None:
        return None
    return (dt - release_dt).total_seconds() / 86400.0


def _finite_float(value):
    """Return value as a finite float, or None if it is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        try:
            f = float(value.strip())
        except (ValueError, TypeError):
            return None
        return f if math.isfinite(f) else None
    return None


# ---------------------------------------------------------------------------
# Tracked-players curve N(t) and cohort-survival earners
# ---------------------------------------------------------------------------

class PlayerCurve:
    """Base curve: N(t) plus cached cohort prefix sums for earners()."""

    def __init__(self):
        self._cohort_cache = {}
        self._w_cache = {}

    def n(self, t):
        raise NotImplementedError

    def cohorts(self, t):
        """(join_times, sizes, prefix_cumulative_sizes) for time t."""
        cached = self._cohort_cache.get(t)
        if cached is not None:
            return cached
        if t <= 0:
            result = ([], [], [0.0])
            self._cohort_cache[t] = result
            return result
        steps = max(24, int(t * 24))
        ds = t / steps
        times = []
        sizes = []
        prefix = [0.0]
        prev_n = self.n(0.0)
        for k in range(steps):
            nxt = self.n((k + 1) * ds)
            size = nxt - prev_n
            if size < 0.0:
                size = 0.0
            times.append((k + 0.5) * ds)
            sizes.append(size)
            prefix.append(prefix[-1] + size)
            prev_n = nxt
        result = (times, sizes, prefix)
        self._cohort_cache[t] = result
        return result

    def prefix_w(self, t, tau):
        """Cumulative sum of size * exp(-(t - s) / tau) over cohorts."""
        key = (t, tau)
        cached = self._w_cache.get(key)
        if cached is not None:
            return cached
        times, sizes, _prefix = self.cohorts(t)
        out = [0.0]
        total = 0.0
        for s, size in zip(times, sizes):
            total += size * math.exp(-(t - s) / tau)
            out.append(total)
        self._w_cache[key] = out
        return out

    def earners(self, t, d, a_eff, tau, e0):
        """Earners at time t for a trophy with delay d and amplitude a_eff."""
        total = float(e0)
        if t <= 0 or a_eff <= 0 or tau <= 0:
            return total
        if t <= d:
            return total
        age = t - d
        # Launch cohort (638 players present at launch).
        if age / tau > 700.0:
            launch_prob = a_eff
        else:
            launch_prob = a_eff * (1.0 - math.exp(-age / tau))
        spare = 1.0 - e0 / 638.0
        if spare > 0.0:
            total += 638.0 * launch_prob * spare
        # Join cohorts.
        times, sizes, prefix_c = self.cohorts(t)
        if not times:
            return total
        if d / tau > 60.0:
            # Overflow guard: never compute exp(d / tau). Cohorts that
            # joined at or before t - d - 40*tau contribute in full;
            # only the narrow boundary band uses the exact exponential.
            full_until = t - d - 40.0 * tau
            idx_full = bisect.bisect_right(times, full_until)
            join = prefix_c[idx_full]
            idx_end = bisect.bisect_left(times, t - d)
            for k in range(idx_full, idx_end):
                s = times[k]
                join += sizes[k] * (1.0 - math.exp(-(t - s - d) / tau))
            total += a_eff * join
        else:
            u = t - d
            idx = bisect.bisect_left(times, u)
            c_val = prefix_c[idx]
            w_val = self.prefix_w(t, tau)[idx]
            total += a_eff * (c_val - math.exp(d / tau) * w_val)
        return total


class AnchorCurve(PlayerCurve):
    """Piecewise log-linear N(t) through anchor points."""

    def __init__(self, anchors):
        super().__init__()
        pts = sorted((float(t), float(n)) for t, n in anchors if float(n) > 0)
        self.anchors = pts

    def n(self, t):
        pts = self.anchors
        if not pts:
            return 638.0
        if len(pts) == 1:
            return pts[0][1]
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            t0, n0 = pts[-2]
            t1, n1 = pts[-1]
            slope = (math.log(n1) - math.log(n0)) / (t1 - t0)
            return math.exp(math.log(n1) + slope * (t - t1))
        for k in range(len(pts) - 1):
            t0, n0 = pts[k]
            t1, n1 = pts[k + 1]
            if t0 <= t <= t1:
                if t1 == t0:
                    return n1
                frac = (t - t0) / (t1 - t0)
                return math.exp(math.log(n0) + frac * (math.log(n1) - math.log(n0)))
        return pts[-1][1]


class MaxCurve(PlayerCurve):
    """Pointwise max of two curves."""

    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def n(self, t):
        a = self.first.n(t)
        b = self.second.n(t)
        return a if a >= b else b


# ---------------------------------------------------------------------------
# Solving for the per-trophy delay d and effective amplitude A_eff
# ---------------------------------------------------------------------------

def solve_delay(curve, t_cal, n_cal, e0, target, a_centre, tau):
    """Bisect (20 iterations) for (d, A_eff) matching `target` at t_cal.

    rate(d, A) = earners(t_cal, d, A, tau, e0) / n_cal * 100.
    If rate(0, A_centre) <= target, raise A on [A_centre, 1.0];
    otherwise raise d on [0, 7].
    """
    if n_cal <= 0:
        n_cal = 638.0

    def rate(d, a_val):
        return curve.earners(t_cal, d, a_val, tau, e0) / n_cal * 100.0

    r0 = rate(0.0, a_centre)
    if r0 <= target:
        if rate(0.0, 1.0) <= target:
            return (0.0, 1.0)
        lo = a_centre
        hi = 1.0
        for _ in range(20):
            mid = (lo + hi) / 2.0
            if rate(0.0, mid) < target:
                lo = mid
            else:
                hi = mid
        return (0.0, (lo + hi) / 2.0)
    lo = 0.0
    hi = 7.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if rate(mid, a_centre) > target:
            lo = mid
        else:
            hi = mid
    return ((lo + hi) / 2.0, a_centre)


# ---------------------------------------------------------------------------
# Method A and Method C
# ---------------------------------------------------------------------------

def method_a(trophies, launch_rates, curve, t_target, horizon_minutes):
    n_target = curve.n(t_target)
    if n_target <= 0:
        n_target = 638.0
    n_24 = curve.n(1.0)
    if n_24 <= 0:
        n_24 = 638.0
    sigma_scale = math.sqrt(max(0.0, t_target))
    short_horizon = horizon_minutes < 720
    out = []
    for idx, tid in enumerate(trophies):
        cls = trophy_class(tid)
        a_c, tau_c = CENTRES_A[cls]
        launch = launch_rates[idx]
        e0 = int(round(launch / 100.0 * 638.0))
        target24 = launch * obs_ratio(tid)
        d, a_eff = solve_delay(curve, 1.0, n_24, e0, target24, a_c, tau_c)
        if short_horizon:
            e_t = e0 + (n_target - 638.0) * JOIN_FRAC[cls] * (e0 / 638.0)
            median = e_t / n_target * 100.0
        else:
            e_t = curve.earners(t_target, d, a_eff, tau_c, e0)
            median = e_t / n_target * 100.0
        sigma = max(SPREAD_SHORT, SIGMA_CLASS[cls] * sigma_scale * SPREAD_MULT)
        base = logit(median)
        out.append([inv_logit(base + z * sigma) for z in Z_SCORES])
    return out


def method_c(trophies, launch_rates, last_rates, growth, t_last,
             cal_time, cal_targets, t_target):
    # One scenario list per trophy: list of (value, weight).
    scenarios = [[] for _ in trophies]
    curve_cache = {}
    for n24 in GRID_N24:
        for nfin in GRID_NFIN:
            key = (n24, nfin)
            curve = curve_cache.get(key)
            if curve is None:
                curve = AnchorCurve([(0.0, 638.0), (1.0, n24), (7.0, nfin)])
                curve_cache[key] = curve
            n_cal = curve.n(cal_time)
            if n_cal <= 0:
                n_cal = 638.0
            n_tgt = curve.n(t_target)
            if n_tgt <= 0:
                n_tgt = 638.0
            for a_mult, a_w in GRID_A_MULT:
                for tau_mult, tau_w in GRID_TAU_MULT:
                    w = 0.04 * a_w * tau_w
                    for idx, tid in enumerate(trophies):
                        cls = trophy_class(tid)
                        a_c, tau_c = CENTRES[cls]
                        a_s = a_c * a_mult
                        tau_s = tau_c * tau_mult
                        e0 = int(round(launch_rates[idx] / 100.0 * 638.0))
                        d, a_eff = solve_delay(
                            curve, cal_time, n_cal, e0,
                            cal_targets[idx], a_s, tau_s)
                        e_t = curve.earners(t_target, d, a_eff, tau_s, e0)
                        scenarios[idx].append((e_t / n_tgt * 100.0, w))
    out = []
    for idx in range(len(trophies)):
        pairs = sorted(scenarios[idx], key=lambda p: p[0])
        total_w = sum(w for _v, w in pairs)
        row = []
        for level in LEVELS:
            threshold = level * total_w
            acc = 0.0
            chosen = pairs[-1][0] if pairs else 0.0
            for value, w in pairs:
                acc += w
                if acc >= threshold:
                    chosen = value
                    break
            row.append(clamp(chosen))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Case handling
# ---------------------------------------------------------------------------

def _read_timestamp(read, release_dt):
    for field in ("at", "slot"):
        dt = parse_time(read.get(field)) if isinstance(read, dict) else None
        if dt is not None:
            return days_since_launch(dt, release_dt)
    return None


def forecast_case(case):
    if not isinstance(case, dict):
        case = {}
    trophies_raw = case.get("trophies")
    trophies = []
    if isinstance(trophies_raw, list):
        for entry in trophies_raw:
            if isinstance(entry, dict):
                tid = _finite_float(entry.get("id"))
                if tid is not None:
                    trophies.append(int(tid))
    history = case.get("history")
    if not isinstance(history, list):
        history = []
    horizon = _finite_float(case.get("horizon_minutes"))
    horizon_minutes = horizon if horizon is not None else 0.0

    game = case.get("game")
    release_dt = LAUNCH_DT
    if isinstance(game, dict):
        parsed = parse_time(game.get("release_date"))
        if parsed is not None:
            release_dt = parsed

    target_dt = parse_time(case.get("target"))
    at_dt = parse_time(case.get("at"))
    if target_dt is not None:
        t_target = days_since_launch(target_dt, release_dt)
    elif at_dt is not None:
        base = days_since_launch(at_dt, release_dt)
        t_target = base + horizon_minutes / 1440.0 if base is not None else None
    else:
        t_target = horizon_minutes / 1440.0
    if t_target is None or t_target < 0:
        t_target = 0.0

    n_trophies = len(trophies)
    launch_rates = [0.0] * n_trophies
    last_rates = [0.0] * n_trophies
    # Launch/last rates with explicit seen flags so a genuine first rate
    # of 0.0 is not mistaken for "missing".
    launch_rates = [0.0] * n_trophies
    last_rates = [0.0] * n_trophies
    seen = [False] * n_trophies
    for read in history:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            continue
        for i in range(min(n_trophies, len(rates))):
            val = _finite_float(rates[i])
            if val is None:
                continue
            if not seen[i]:
                launch_rates[i] = val
                seen[i] = True
            last_rates[i] = val

    # No history at all: persistence baseline for both methods.
    if not history:
        spread = 0.5 * math.sqrt(max(0.0, horizon_minutes) / 1440.0)
        rows = []
        for i, tid in enumerate(trophies):
            base = logit(last_rates[i])
            qs = [inv_logit(base + z * spread) for z in Z_SCORES]
            rows.append({"id": tid, "q": _finalise(qs)})
        return {"trophies": rows}

    # Observed players.
    observed = {}
    for read in history:
        if not isinstance(read, dict):
            continue
        n_val = _finite_float(read.get("players_tracked"))
        if n_val is None or n_val <= 0:
            continue
        t_read = _read_timestamp(read, release_dt)
        if t_read is None:
            continue
        if t_read not in observed or n_val > observed[t_read]:
            observed[t_read] = n_val
    obs_points = sorted(observed.items())
    growth = False
    if obs_points:
        first_n = obs_points[0][1]
        last_n = obs_points[-1][1]
        growth = last_n > 1.5 * first_n

    t_last = None
    if history:
        last_read = history[-1]
        if isinstance(last_read, dict):
            t_last = _read_timestamp(last_read, release_dt)

    # Method A curve.
    curve_a = AnchorCurve(ANCHORS_A)
    if growth and obs_points:
        curve_a = MaxCurve(curve_a, AnchorCurve(obs_points))

    q_a = method_a(trophies, launch_rates, curve_a, t_target, horizon_minutes)

    # Method C calibration.
    if growth and t_last is not None and t_last > 0:
        cal_time = t_last
        cal_targets = list(last_rates)
    else:
        cal_time = 1.0
        cal_targets = [launch_rates[i] * obs_ratio(trophies[i])
                       for i in range(n_trophies)]
    q_c = method_c(trophies, launch_rates, last_rates, growth, t_last,
                   cal_time, cal_targets, t_target)

    rows = []
    for i, tid in enumerate(trophies):
        blended = []
        for k in range(5):
            mixed = 0.5 * logit(q_a[i][k]) + 0.5 * logit(q_c[i][k])
            blended.append(clamp(inv_logit(mixed)))
        rows.append({"id": tid, "q": _finalise(blended)})
    return {"trophies": rows}


def _finalise(values):
    ordered = sorted(values)
    rounded = [round4(v) for v in ordered]
    fixed = []
    prev = None
    for v in rounded:
        if prev is not None and v < prev:
            v = prev
        fixed.append(v)
        prev = v
    return fixed


# ---------------------------------------------------------------------------
# Kit contract: *.in -> *.out
# ---------------------------------------------------------------------------

def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    try:
        names = sorted(os.listdir(input_dir))
    except OSError:
        names = []
    for name in names:
        if not name.endswith(".in"):
            continue
        in_path = os.path.join(input_dir, name)
        if not os.path.isfile(in_path):
            continue
        try:
            with open(in_path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
        except (OSError, ValueError):
            case = {}
        try:
            result = forecast_case(case)
        except Exception:
            # Never crash on odd input: emit a minimal valid forecast.
            trophies = []
            raw = case.get("trophies") if isinstance(case, dict) else None
            if isinstance(raw, list):
                for entry in raw:
                    if isinstance(entry, dict):
                        tid = _finite_float(entry.get("id"))
                        if tid is not None:
                            trophies.append({"id": int(tid), "q": [0.01, 0.01, 0.01, 0.01, 0.01]})
            result = {"trophies": trophies}
        stem = name[:-3]
        out_path = os.path.join(output_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

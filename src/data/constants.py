"""Column indices, action codes, vehicle types, solution/config/state indices.

Single source of truth. Units: meters, seconds, grams, VND. All int64 on hot path.
"""
import numpy as np

# ── Customer columns (i64[:, 7]) ──
COL_X = 0           # meters
COL_Y = 1           # meters
COL_DEMAND = 2      # grams
COL_TW_OPEN = 3     # seconds from day start
COL_TW_CLOSE = 4    # seconds
COL_SERVICE = 5     # service time (seconds)
COL_RESTRICTED = 6  # 0=both, 1=bike only
CUST_COLS = 7

# ── Vehicle columns (i64[:, 4]) ──
VCOL_TYPE = 0       # 0=truck, 1=bike
VCOL_CAPACITY = 1   # grams
VCOL_COST_M = 2     # VND per meter
VCOL_SPEED = 3      # microseconds per meter (1e6 * 3600 / (kmh * 1000))

# ── Action codes (i8) ──
ACT_DELIVER = 0
ACT_RELOAD = 1
ACT_PAD = -1

# ── Vehicle types ──
VEH_TRUCK = 0
VEH_BIKE = 1

# ── Satellite columns (i64[:, 5]) ──
SAT_CUST = 0
SAT_BIKE = 1
SAT_TRUCK = 2
SAT_KG = 3          # grams
SAT_TIME = 4        # seconds

# ── Route state columns (simulation output, i64[:,:,10]) ──
ST_CUST = 0
ST_ACTION = 1
ST_ARRIVE = 2       # seconds
ST_WAIT = 3         # seconds
ST_START = 4        # seconds
ST_SERVICE = 5      # seconds
ST_DEPART = 6       # seconds
ST_LOAD_BEF = 7     # grams
ST_LOAD_AFT = 8     # grams
ST_FEASIBLE = 9     # 0 or 1
ST_COLS = 10

# ── Solution tuple indices ──
SOL_TRUCK_STOPS = 0
SOL_TRUCK_ACTIONS = 1
SOL_BIKE_STOPS = 2
SOL_BIKE_ACTIONS = 3
SOL_TRUCK_LENGTHS = 4
SOL_BIKE_LENGTHS = 5
SOL_TRUCK_LOADS = 6
SOL_BIKE_LOADS = 7
SOL_TRUCK_DISTANCES = 8
SOL_BIKE_DISTANCES = 9
SOL_CUST_VEHICLE = 10
SOL_CUST_VTYPE = 11
SOL_CUST_ROUTE_POS = 12
SOL_SATELLITES = 13
SOL_META = 14
SOL_SIZE = 15

# ── Solution meta indices (sol[SOL_META]) ──
META_N_TRUCKS = 0
META_N_BIKES = 1
META_MAX_ROUTE_LEN = 2
META_N_CUSTOMERS = 3
META_N_SATELLITES = 4

# ── Config array indices (f64, tuning only) ──
CFG_MAX_ITERATIONS = 0
CFG_SEGMENT_LENGTH = 1
CFG_NO_IMPROVE_LIMIT = 2
CFG_SA_INITIAL_TEMP = 3
CFG_SA_COOLING_RATE = 4
CFG_SA_MIN_TEMP = 5
CFG_REACTION_FACTOR = 6
CFG_SIGMA_1 = 7
CFG_SIGMA_2 = 8
CFG_SIGMA_3 = 9
CFG_PENALTY_W3_START = 10
CFG_PENALTY_W3_END = 11
CFG_PENALTY_DECAY_RATE = 12
CFG_LS_FREQUENCY = 13
CFG_LS_ON_NEW_BEST = 14
CFG_CROSS_FREQUENCY = 15
CFG_Q_MIN = 16
CFG_Q_MAX_PCT = 17
CFG_Q_MAX = 18
CFG_SHAW_RANDOMNESS = 19
CFG_WORST_NOISE = 20
CFG_ZONE_PCT = 21
CFG_REGRET_K = 22
CFG_REGRET_NOISE = 23
CFG_SAT_THRESHOLD_KM = 24
CFG_REHEAT_THRESHOLD_PCT = 25
CFG_REHEAT_TEMP_PCT = 26
CFG_ELITE_POOL_SIZE = 27
CFG_ELITE_MIN_HAMMING_PCT = 28
CFG_RESTART_THRESHOLD_PCT = 29
CFG_MAX_RESTARTS = 30
CFG_QL_ENABLED = 31
CFG_QL_ALPHA = 32
CFG_QL_GAMMA = 33
CFG_QL_EPSILON = 34
CFG_QL_EPSILON_DECAY = 35
CFG_QL_EPSILON_MIN = 36
CFG_DELTA_T = 37
CFG_BLINK_PROB = 38
CFG_SISR_MAX_STRING_LEN = 39
CFG_SISR_MAX_ROUTES = 40
CFG_HISTORY_NOISE = 41
CFG_SIZE = 42

# ── ALNS state scalar indices (state_f, f64) ──
STF_CURRENT_FITNESS = 0
STF_BEST_FITNESS = 1
STF_BEST_FEAS_FITNESS = 2
STF_TEMPERATURE = 3
STF_INITIAL_TEMP = 4
STF_W3 = 5
STF_NO_IMPROVE_COUNT = 6
STF_RESTART_COUNT = 7
STF_HISTORY_BEST_COUNT = 8
STF_LAST_D_IDX = 9
STF_LAST_R_IDX = 10
STF_LAST_CX_IDX = 11
STF_ITERATION = 12
STF_ACCEPTED = 13
STF_REHEAT_DONE = 14
STF_CROSS_USED = 15
STF_HAS_BEST_FEASIBLE = 16
STF_SIZE = 20

# ── Eval result indices (i64) ──
EV_FITNESS = 0
EV_COST = 1
EV_SYNC_COST = 2
EV_MAKESPAN = 3
EV_TOTAL_PENALTY = 4
EV_TOTAL_WAIT = 5
EV_FEASIBLE = 6      # 0 or 1
EV_SIM_FEASIBLE = 7
EV_VALID = 8
EV_N_FEASIBLE_ROUTES = 9
EV_SIZE = 10

# ── Penalty weight indices (i64 VND) ──
PW_UNSERVED = 0
PW_DUPLICATE = 1
PW_CAPACITY = 2
PW_TIME_WINDOW = 3
PW_SYNC = 4
PW_VEHICLE_RESTRICTION = 5
PW_SIZE = 6

# ── Logger column indices ──
LOG_ITERATION = 0
LOG_FITNESS = 1
LOG_COST = 2
LOG_MAKESPAN = 3
LOG_PENALTY = 4
LOG_FEASIBLE = 5
LOG_ACCEPTED = 6
LOG_TEMPERATURE = 7
LOG_W3 = 8
LOG_DESTROY_OP = 9
LOG_REPAIR_OP = 10
LOG_BEST_FITNESS = 11
LOG_PARETO_SIZE = 12
LOG_COLS = 13

# ── Operator counts ──
N_DESTROY_OPS = 13
N_REPAIR_OPS = 5
N_CROSS_OPS = 5

# ── Default max sizes ──
MAX_SATELLITES = 200
MAX_VIOLATIONS = 500
MAX_ELITE_SIZE = 5
MAX_PARETO_SIZE = 50

# ── Unit conversion helpers ──
def kmh_to_us_per_m(kmh):
    """Convert km/h to microseconds per meter. Python only (cold path)."""
    return int(1_000_000 * 3600 / (kmh * 1000))


from numba import njit as _njit

@_njit(cache=True)
def travel_time_s(dist_m, speed_us_per_m):
    """Travel time in seconds. Integer arithmetic. @njit."""
    return dist_m * speed_us_per_m // 1_000_000

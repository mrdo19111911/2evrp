"""Operating costs — Vietnam 2025-2026. ALL integer, ALL per-meter/per-second.

Units: VND (int64), meters, seconds, grams, microseconds-per-meter.
No float64 anywhere. Change values here to tune the model.
"""

# ============================================================
# TRUCK 2 TON — VND per meter
# ============================================================
TRUCK_FUEL_COST_M      = 3       # VND/m  (3360 VND/km / 1000)
TRUCK_MAINTENANCE_M    = 1       # VND/m
TRUCK_DEPRECIATION_M   = 2       # VND/m  (1500/1000 rounded)
TRUCK_DRIVER_M         = 2       # VND/m  (2250/1000 rounded)
TRUCK_OTHER_M          = 4       # VND/m  (3890/1000 rounded)
TRUCK_COST_PER_M       = 12      # VND/m  (12000 VND/km)

TRUCK_DRIVER_HOUR      = 56250   # VND/hour
TRUCK_DRIVER_SEC       = 16      # VND/s  (56250/3600 ~ 15.6, rounded up)
TRUCK_FIXED_DAY        = 500000  # VND/day
TRUCK_DEPLOY_COST      = 150000  # VND/trip

TRUCK_SPEED_US_PER_M   = 144000  # us/m  (25 km/h)
TRUCK_CAPACITY_G       = 2000000 # grams (2000 kg)

TRUCK_WAIT_COST_SEC    = 49      # VND/s  (2969 VND/min / 60 ~ 49.5)

# ============================================================
# BIKE (XE MAY) — VND per meter
# ============================================================
BIKE_FUEL_COST_M       = 1       # VND/m  (625/1000 rounded)
BIKE_MAINTENANCE_M     = 0       # VND/m  (300/1000 ~ 0)
BIKE_DEPRECIATION_M    = 0       # VND/m
BIKE_DRIVER_M          = 2       # VND/m
BIKE_OTHER_M           = 2       # VND/m
BIKE_COST_PER_M        = 5       # VND/m  (5000 VND/km)

BIKE_DRIVER_HOUR       = 37500   # VND/hour
BIKE_DRIVER_SEC        = 10      # VND/s  (37500/3600 ~ 10.4)
BIKE_FIXED_DAY         = 50000   # VND/day
BIKE_DEPLOY_COST       = 30000   # VND/trip

BIKE_SPEED_US_PER_M    = 180000  # us/m  (20 km/h)
BIKE_CAPACITY_G        = 60000   # grams (60 kg)

BIKE_WAIT_COST_SEC     = 19      # VND/s  (1146 VND/min / 60 ~ 19.1)

# ============================================================
# SERVICE / RELOAD — seconds (i64)
# ============================================================
RELOAD_SERVICE_TIME    = 300     # seconds (5 min)
RELOAD_HANDLING_COST   = 5000    # VND / reload event

# ============================================================
# TIME — seconds (i64)
# ============================================================
DAY_START              = 0       # seconds
DAY_LENGTH             = 28800   # seconds (8 hours)
SYNC_DELTA_T           = 900     # seconds (15 min)

# ============================================================
# PENALTIES — VND (i64)
# ============================================================
PENALTY_UNSERVED       = 500000
PENALTY_UNSERVED_MULTIPLIER = 5  # integer multiplier
PENALTY_LATE           = 167     # VND/s  (10000 VND/min / 60)
PENALTY_OVERLOAD_PER_PCT = 10    # per 1% overload * fixed_cost
PENALTY_SYNC_FAIL      = 100000
PENALTY_RESTRICTION    = 200000
PENALTY_OVERTIME       = 1667    # VND/s  (100000 VND/min / 60)
PENALTY_MISSING_RELOAD = 1000000
PENALTY_SYNC_GAP_SEC   = 333     # VND/s  (20000 VND/min / 60)

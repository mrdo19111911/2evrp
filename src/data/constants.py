"""Column indices, action codes, vehicle types. Single source of truth."""
import numpy as np

# Customer columns
COL_X = 0
COL_Y = 1
COL_DEMAND = 2
COL_TW_OPEN = 3
COL_TW_CLOSE = 4
COL_SERVICE_TIME = 5

# Vehicle columns
VCOL_TYPE = 0
VCOL_CAPACITY = 1
VCOL_COST_KM = 2
VCOL_SPEED = 3

# Action codes
ACT_DELIVER = 0
ACT_RELOAD = 1
ACT_PAD = -1

# Vehicle types
VEH_TRUCK = 0
VEH_BIKE = 1

# Satellite columns
SAT_CUST = 0
SAT_BIKE = 1
SAT_TRUCK = 2
SAT_KG = 3
SAT_TIME = 4

# Route state columns (simulation output)
ST_CUST = 0
ST_ACTION = 1
ST_ARRIVE = 2
ST_WAIT = 3
ST_START = 4
ST_SERVICE = 5
ST_DEPART = 6
ST_LOAD_BEF = 7
ST_LOAD_AFT = 8
ST_FEASIBLE = 9
ST_COLS = 10

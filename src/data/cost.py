"""Realistic operating costs — Vietnam 2025-2026.
ALL parameters configurable. Change values here to tune the model.

Sources:
- Truck fuel: https://otophucuong.vn/dinh-muc-tieu-hao-nguyen-lieu-xe-tai/
- Bike fuel: https://theicct.org/wp-content/uploads/2025/02/ID-250-Vietnam-two-wheelers_report_final.pdf
- Freight rates: https://247express.vn/tin-tuc/logistics-44/gia-van-tai-duong-bo/535
- Fuel prices: https://rentabikevn.com/petrol-in-vietnam-all-you-need-to-know/
"""

# ============================================================
# TRUCK 2 TON — Chi phi van hanh
# ============================================================
TRUCK_FUEL_CONSUMPTION = 16.0       # lit / 100 km (do thi, co tai, stop-and-go)
TRUCK_FUEL_PRICE       = 21000      # VND / lit (diesel 2025)
TRUCK_FUEL_COST_KM     = 3360       # VND / km (= 16 * 21000 / 100)
TRUCK_MAINTENANCE_KM   = 1000       # VND / km (bao tri, lop, dau may)
TRUCK_DEPRECIATION_KM  = 1500       # VND / km (khau hao ~550M / 350k km)
TRUCK_DRIVER_KM        = 2250       # VND / km (= 56250 / 25 km/h = driver cost per km)
TRUCK_OTHER_KM         = 3890       # VND / km (toll, bao hiem, phi duong bo)
TRUCK_TOTAL_KM         = 12000      # VND / km (= 3360 + 1000 + 1500 + 2250 + 3890)

TRUCK_DRIVER_HOUR      = 56250      # VND / gio (~9M/thang / 160h)
TRUCK_FIXED_DAY        = 500000     # VND / ngay (bao hiem, phi duong bo, dau xe, toll)
TRUCK_DEPLOY_COST      = 150000     # VND / chuyen (chi phi khoi hanh: kiem tra xe, giay to, xuat kho)

TRUCK_SPEED_URBAN      = 25.0       # km/h (do thi, giao hang)
TRUCK_SPEED_SUBURBAN   = 40.0       # km/h (ngoai thanh)
TRUCK_CAPACITY         = 2000.0     # kg

# ============================================================
# BIKE (XE MAY) — Chi phi van hanh
# ============================================================
BIKE_FUEL_CONSUMPTION  = 2.5        # lit / 100 km (do thi, giao hang)
BIKE_FUEL_PRICE        = 25000      # VND / lit (RON 95, 2025)
BIKE_FUEL_COST_KM      = 625        # VND / km (= 2.5 * 25000 / 100)
BIKE_MAINTENANCE_KM    = 300        # VND / km
BIKE_DEPRECIATION_KM   = 175        # VND / km (~35M / 200k km)
BIKE_DRIVER_KM         = 1875       # VND / km (= 37500 / 20 km/h = driver cost per km)
BIKE_OTHER_KM          = 2025       # VND / km (bao hiem, phu cap)
BIKE_TOTAL_KM          = 5000       # VND / km (= 625 + 300 + 175 + 1875 + 2025)

BIKE_DRIVER_HOUR       = 37500      # VND / gio (~6M/thang / 160h, shipper)
BIKE_FIXED_DAY         = 50000      # VND / ngay (bao hiem)
BIKE_DEPLOY_COST       = 30000      # VND / chuyen (chi phi khoi hanh)

BIKE_SPEED_URBAN       = 20.0       # km/h (do thi, vao hem)
BIKE_SPEED_SUBURBAN    = 35.0       # km/h
BIKE_CAPACITY          = 60.0       # kg

# ============================================================
# WAITING COST — Chi phi cho doi
# ============================================================
# Chi phi cho = 50% chi phi di chuyen trong thoi gian do
# Ly do: xe dung, van ton driver salary + co hoi
WAIT_COST_RATIO        = 0.5        # 50% of travel cost rate

# Truck: travel cost/min = 12000 * 25 / 60 + 56250 / 60 = 5000 + 938 = 5938 VND/min
# Wait cost/min = 5938 * 0.5 = 2969 VND/min
TRUCK_WAIT_COST_MIN    = 2969       # VND / phut cho doi

# Bike: travel cost/min = 5000 * 20 / 60 + 37500 / 60 = 1667 + 625 = 2292 VND/min
# Wait cost/min = 2292 * 0.5 = 1146 VND/min
BIKE_WAIT_COST_MIN     = 1146       # VND / phut cho doi

# ============================================================
# SERVICE TIME — Thoi gian giao hang
# ============================================================
# service_time = SERVICE_BASE + SERVICE_PER_100KG * (demand / 100)
# VD: 50kg → 10 + 5*0.5 = 12.5 phut
#     200kg → 10 + 5*2.0 = 20 phut
SERVICE_BASE           = 10.0       # phut co dinh (do xe, ky nhan, ...)
SERVICE_PER_100KG      = 5.0        # phut / 100 kg (boc do)

# ============================================================
# RELOAD — Chi phi trung chuyen
# ============================================================
RELOAD_SERVICE_TIME    = 5.0        # phut / lan transfer
RELOAD_HANDLING_COST   = 5000       # VND / lan (cong boc do)

# ============================================================
# MULTIPLE TRIPS
# ============================================================
# Moi xe chi chay 1 chuyen/ngay: depot -> customers -> depot
# Tong thoi gian phai <= DAY_LENGTH. Khong reload tai depot.

# ============================================================
# TIME — Thoi gian lam viec
# ============================================================
DAY_START              = 0.0        # phut (7:00 AM = minute 0)
DAY_LENGTH             = 480.0      # phut (8 gio lam viec)
SYNC_DELTA_T           = 15.0       # phut (tolerance dong bo truck-bike)

# ============================================================
# PENALTIES — Phat vi pham (VND)
# ============================================================
PENALTY_UNSERVED       = 500000     # default fallback, thuc te tinh DYNAMIC per customer
PENALTY_UNSERVED_MULTIPLIER = 5.0  # penalty = X lan chi phi chuyen re nhat (bike round trip)
PENALTY_LATE           = 10000      # / phut tre deadline
PENALTY_OVERLOAD_PER_PCT = 10       # per 1% overload: fixed_cost * 10 (linear)
# Legacy (kept for reference):
PENALTY_OVERLOAD_KG    = 50000      # / kg vuot tai (0-10kg) — DEPRECATED
PENALTY_OVERLOAD_KG_HEAVY = 100000  # / kg vuot tai (>10kg) — DEPRECATED
PENALTY_OVERLOAD_THRESHOLD = 10.0   # kg: moc tang penalty — DEPRECATED
PENALTY_SYNC_FAIL      = 100000     # / lan sync that bai
PENALTY_RESTRICTION    = 200000     # / lan vi pham xe vao hem
PENALTY_OVERTIME       = 100000     # / phut vuot qua DAY_LENGTH (phai du manh de force drop)
PENALTY_MISSING_RELOAD = 1000000    # bike reload tai node ma truck KHONG den
PENALTY_SYNC_GAP_MIN   = 20000      # VND / phut vuot epsilon (sync gap)

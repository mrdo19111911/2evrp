"""Quick import check for all modules after int64 refactor."""
import sys

modules = [
    ("data.constants", "from src.data.constants import COL_RESTRICTED, COL_SERVICE, VCOL_COST_M, CUST_COLS, kmh_to_us_per_m"),
    ("data.cost", "from src.data.cost import DAY_LENGTH, RELOAD_SERVICE_TIME"),
    ("solution.structure", "from src.solution.structure import create_solution, copy_solution_into"),
    ("solution._helpers", "from src.solution._helpers import get_loads, get_distances, update_route_distance"),
    ("solution.route_ops", "from src.solution.route_ops import insert_stop, remove_stop"),
    ("solution.query", "from src.solution.query import get_assigned_customers, get_customer_info"),
    ("solution.check", "from src.solution.check import can_insert_customer, can_swap_customers"),
    ("solution.delta", "from src.solution.delta import insertion_cost_delta, find_best_insertion_all_routes"),
    ("solution.satellite_ops", "from src.solution.satellite_ops import add_satellite_event, remove_satellites_for_customer"),
    ("engine.simulate", "from src.engine.simulate import simulate_route_into, simulate_all_routes"),
    ("engine.route_cost", "from src.engine.route_cost import compute_route_cost, compute_makespan"),
    ("engine.reload", "from src.engine.reload import truck_reload_at_stop, bike_reload_at_stop"),
    ("engine.validate", "from src.engine.validate import validate_all"),
    ("engine.violations", "from src.engine.violations import compute_penalties"),
    ("engine.fitness", "from src.engine.fitness import evaluate_solution"),
    ("engine.sync_cost", "from src.engine.sync_cost import compute_sync_cost"),
    ("alns.config", "from src.alns.config import make_config"),
    ("alns.destroy", "from src.alns.destroy import DESTROY_OPS"),
    ("alns.repair", "from src.alns.repair import REPAIR_OPS"),
    ("alns.crosslayer", "from src.alns.crosslayer import cross_dispatch"),
    ("alns.local_search", "from src.alns.local_search import run_local_search"),
    ("alns.ls_intra", "from src.alns.ls_intra import two_opt, or_opt"),
    ("alns.ls_inter", "from src.alns.ls_inter import relocate_inter_route"),
    ("alns.ls_twooptstar", "from src.alns.ls_twooptstar import two_opt_star"),
    ("alns.ls_advanced", "from src.alns.ls_advanced import swap_star, ejection_chain_3"),
    ("alns.loop", "from src.alns.loop import solve"),
    ("alns.setup", "from src.alns.setup import init_search_state"),
    ("data.io", "from src.data.io import load_instance"),
    ("data.distance", "from src.data.distance import compute_dist_matrix"),
    ("init.builder", "from src.init.builder import build_initial_solution"),
]

ok, fail = 0, 0
for name, stmt in modules:
    try:
        exec(stmt)
        print(f"  OK  {name}")
        ok += 1
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        fail += 1

print(f"\n{ok} OK, {fail} FAIL")
sys.exit(1 if fail > 0 else 0)

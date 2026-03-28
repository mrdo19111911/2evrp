"""Local search — orchestrator. @njit eliminates Python dispatch overhead."""
from numba import njit
from src.data.constants import VEH_TRUCK, VEH_BIKE, SOL_META, META_N_TRUCKS, META_N_BIKES
from src.alns.ls_intra import two_opt, or_opt
from src.alns.ls_inter import relocate_inter_route, exchange_inter_route
from src.alns.ls_twooptstar import two_opt_star
from src.alns.ls_advanced import swap_star, ejection_chain_3


@njit(cache=True)
def run_local_search(sol, dist_matrix, customers, vehicles):
    """Run all LS operators. Fully compiled — zero Python overhead."""
    n_trucks = sol[SOL_META][META_N_TRUCKS]
    n_bikes = sol[SOL_META][META_N_BIKES]
    improved = False

    for t in range(n_trucks):
        if two_opt(sol, VEH_TRUCK, t, dist_matrix):
            improved = True
        if or_opt(sol, VEH_TRUCK, t, dist_matrix):
            improved = True
    for b in range(n_bikes):
        if two_opt(sol, VEH_BIKE, b, dist_matrix):
            improved = True
        if or_opt(sol, VEH_BIKE, b, dist_matrix):
            improved = True

    # Inter-route: trucks
    if relocate_inter_route(sol, VEH_TRUCK, dist_matrix, customers, vehicles):
        improved = True
    if two_opt_star(sol, VEH_TRUCK, dist_matrix, customers, vehicles):
        improved = True
    if exchange_inter_route(sol, VEH_TRUCK, dist_matrix, customers):
        improved = True
    if swap_star(sol, VEH_TRUCK, dist_matrix, customers):
        improved = True
    # Inter-route: bikes
    if relocate_inter_route(sol, VEH_BIKE, dist_matrix, customers, vehicles):
        improved = True
    if two_opt_star(sol, VEH_BIKE, dist_matrix, customers, vehicles):
        improved = True
    if exchange_inter_route(sol, VEH_BIKE, dist_matrix, customers):
        improved = True
    if swap_star(sol, VEH_BIKE, dist_matrix, customers):
        improved = True

    # Advanced
    if ejection_chain_3(sol, VEH_TRUCK, dist_matrix, customers, vehicles):
        improved = True
    if ejection_chain_3(sol, VEH_BIKE, dist_matrix, customers, vehicles):
        improved = True

    return improved

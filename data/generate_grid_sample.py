"""Generate grid 20x20 sample instance with DM + TM."""
import numpy as np
import json

# === Config ===
GRID_W, GRID_H = 20, 20
CELL_SIZE_KM = 2.0
DAY_LENGTH = 480.0
TRUCK_SPEED = 25.0
BIKE_SPEED = 20.0
N_TRUCKS = 5
N_BIKES = 15


def generate_customers():
    rows = []
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            x = gx * CELL_SIZE_KM
            y = gy * CELL_SIZE_KM
            heavy = (gx % 4 == 0) and (gy % 4 == 0)
            demand = 400.0 if heavy else 10.0
            service_time = 15.0 if heavy else 5.0
            tw_open = (gx + gy) % 16 * 30.0
            tw_width = 120.0 if heavy else 60.0
            tw_close = min(tw_open + tw_width, DAY_LENGTH)
            restricted = 1 if (not heavy and gx % 3 == 1 and gy % 2 == 1) else 0
            rows.append([x, y, demand, tw_open, tw_close, service_time, restricted])
    return np.array(rows, dtype=np.float64)


def compute_dist_matrix(depot, customers_xy):
    coords = np.vstack([depot, customers_xy])
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def compute_time_matrix(dist_matrix, speed_kmh):
    return dist_matrix / speed_kmh * 60.0  # minutes


def make_vehicles():
    vehicles = []
    for i in range(N_TRUCKS):
        vehicles.append({"id": i, "type": "truck", "capacity_kg": 2000.0,
                         "cost_per_km": 4.52, "speed_kmh": TRUCK_SPEED,
                         "cost_per_hour": 37500})
    for i in range(N_BIKES):
        vehicles.append({"id": N_TRUCKS + i, "type": "bike", "capacity_kg": 60.0,
                         "cost_per_km": 0.85, "speed_kmh": BIKE_SPEED,
                         "cost_per_hour": 25000})
    return vehicles


def print_stats(customers, depot):
    n = len(customers)
    heavy = customers[:, 2] > 60
    restricted = customers[:, 6] == 1
    print(f"Grid: {GRID_W}x{GRID_H} = {n} customers")
    print(f"Cell size: {CELL_SIZE_KM} km")
    print(f"Area: {(GRID_W-1)*CELL_SIZE_KM:.0f} x {(GRID_H-1)*CELL_SIZE_KM:.0f} km")
    print(f"Depot: ({depot[0]}, {depot[1]})")
    print(f"\nHeavy (400kg): {heavy.sum()}")
    print(f"Light (10kg):  {(~heavy).sum()}")
    print(f"Restricted:    {restricted.sum()}")
    print(f"\nTotal demand:  {customers[:, 2].sum():.0f} kg")
    print(f"Truck cap:     {N_TRUCKS} x 2000 = {N_TRUCKS * 2000} kg")
    print(f"Bike cap:      {N_BIKES} x 60 = {N_BIKES * 60} kg")


def save_json(customers, depot, dist_matrix, tm_truck, tm_bike, filepath):
    cust_list = []
    for i in range(len(customers)):
        cust_list.append({
            "id": i,
            "x": customers[i, 0],
            "y": customers[i, 1],
            "demand_kg": customers[i, 2],
            "tw_open": customers[i, 3],
            "tw_close": customers[i, 4],
            "service_time": customers[i, 5],
            "restricted": bool(customers[i, 6]),
        })

    output = {
        "metadata": {
            "name": "grid_20x20_5km",
            "description": "20x20 grid, 5km spacing, heavy nodes at x%4==0 & y%4==0",
            "n_customers": len(customers),
            "n_trucks": N_TRUCKS,
            "n_bikes": N_BIKES,
            "area_km2": ((GRID_W - 1) * CELL_SIZE_KM) ** 2,
            "total_demand_kg": float(customers[:, 2].sum()),
        },
        "depot": {"x": float(depot[0]), "y": float(depot[1])},
        "customers": cust_list,
        "vehicles": {
            "trucks": [v for v in make_vehicles() if v["type"] == "truck"],
            "bikes": [v for v in make_vehicles() if v["type"] == "bike"],
        },
        "distance_matrix": {
            "description": "Euclidean km. Index 0=depot, 1..N=customers.",
            "shape": list(dist_matrix.shape),
            "data": dist_matrix.round(4).tolist(),
        },
        "time_matrix": {
            "truck": {
                "speed_kmh": TRUCK_SPEED,
                "shape": list(tm_truck.shape),
                "data": tm_truck.round(4).tolist(),
            },
            "bike": {
                "speed_kmh": BIKE_SPEED,
                "shape": list(tm_bike.shape),
                "data": tm_bike.round(4).tolist(),
            },
        },
        "config": {
            "day_length_min": DAY_LENGTH,
            "sync_delta_t_min": 15.0,
            "reload_service_time_min": 5.0,
            "reload_handling_cost_vnd": 5000,
        },
    }

    with open(filepath, "w") as f:
        json.dump(output, f)
    print(f"Saved JSON: {filepath} ({os.path.getsize(filepath) / 1e6:.1f} MB)")


def save_npz(customers, depot, vehicles_np, dist_matrix, tm_truck, tm_bike, filepath):
    np.savez_compressed(filepath,
                        customers=customers[:, :6].astype(np.float64),
                        restricted=customers[:, 6].astype(np.int8),
                        depot=depot.astype(np.float64),
                        vehicles=vehicles_np.astype(np.float64),
                        dist_matrix=dist_matrix.astype(np.float32),
                        tm_truck=tm_truck.astype(np.float32),
                        tm_bike=tm_bike.astype(np.float32))
    print(f"Saved NPZ: {filepath} ({os.path.getsize(filepath) / 1e6:.1f} MB)")


if __name__ == "__main__":
    import os

    customers = generate_customers()
    depot = np.array([GRID_W / 2 * CELL_SIZE_KM, GRID_H / 2 * CELL_SIZE_KM])
    print_stats(customers, depot)

    # Distance matrix
    print("\nComputing distance matrix...")
    dm = compute_dist_matrix(depot, customers[:, :2])
    print(f"DM shape: {dm.shape}, min={dm.min():.1f}, max={dm.max():.1f} km")

    # Time matrices
    print("Computing time matrices...")
    tm_truck = compute_time_matrix(dm, TRUCK_SPEED)
    tm_bike = compute_time_matrix(dm, BIKE_SPEED)
    print(f"TM truck: max={tm_truck.max():.1f} min")
    print(f"TM bike:  max={tm_bike.max():.1f} min")

    # Verify: DM[0, 1] = depot → customer 0
    c0 = customers[0, :2]
    expected_dist = np.sqrt((depot[0] - c0[0])**2 + (depot[1] - c0[1])**2)
    assert abs(dm[0, 1] - expected_dist) < 1e-6, "DM sanity check failed"
    print(f"\nSanity: depot->C0 = {dm[0, 1]:.2f} km (expected {expected_dist:.2f})")
    print(f"Sanity: truck time depot->C0 = {tm_truck[0, 1]:.1f} min")
    print(f"Sanity: bike time depot->C0 = {tm_bike[0, 1]:.1f} min")

    # Vehicles as numpy array
    vehicles_np = np.array(
        [[0, 2000.0, 4.52, TRUCK_SPEED]] * N_TRUCKS +
        [[1, 60.0, 0.85, BIKE_SPEED]] * N_BIKES,
        dtype=np.float64
    )

    # Save
    print()
    save_json(customers, depot, dm, tm_truck, tm_bike,
              "e:/2evrp/data/grid_20x20.json")
    save_npz(customers, depot, vehicles_np, dm, tm_truck, tm_bike,
             "e:/2evrp/data/grid_20x20.npz")

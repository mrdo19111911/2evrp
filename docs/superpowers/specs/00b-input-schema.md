# Input Data Schema

## JSON Structure

```json
{
  "metadata": {
    "name": "grid_20x20_5km",
    "description": "20x20 grid, 5km spacing, heavy/light mixed",
    "n_customers": 400,
    "n_trucks": 5,
    "n_bikes": 15,
    "area_km2": 9025,
    "total_demand_kg": 13750,
    "created": "2026-03-28"
  },

  "depot": {
    "x": 50.0,
    "y": 50.0
  },

  "customers": [
    {
      "id": 0,
      "x": 0.0,
      "y": 0.0,
      "demand_kg": 400.0,
      "tw_open": 0.0,
      "tw_close": 120.0,
      "service_time": 15.0,
      "restricted": false,
      "tags": ["heavy"]
    }
  ],

  "vehicles": {
    "trucks": [
      {
        "id": 0,
        "capacity_kg": 2000.0,
        "cost_per_km": 4.52,
        "speed_kmh": 25.0,
        "cost_per_hour": 37500
      }
    ],
    "bikes": [
      {
        "id": 0,
        "capacity_kg": 60.0,
        "cost_per_km": 0.85,
        "speed_kmh": 20.0,
        "cost_per_hour": 25000
      }
    ]
  },

  "distance_matrix": {
    "description": "Euclidean distance (km). Index 0=depot, 1..N=customers.",
    "shape": [401, 401],
    "data": [[0.0, 70.71, ...], ...]
  },

  "time_matrix": {
    "description": "Travel time (minutes). Separate for truck/bike due to different speeds.",
    "truck": {
      "shape": [401, 401],
      "speed_kmh": 25.0,
      "data": [[0.0, 169.7, ...], ...]
    },
    "bike": {
      "shape": [401, 401],
      "speed_kmh": 20.0,
      "data": [[0.0, 212.1, ...], ...]
    }
  },

  "config": {
    "day_length_min": 480,
    "sync_delta_t_min": 15,
    "reload_service_time_min": 5,
    "reload_handling_cost_vnd": 5000
  }
}
```

## Notes

- **distance_matrix**: DM[i][j] = khoảng cách km giữa node i và j. i=0 là depot.
- **time_matrix**: TM riêng cho truck/bike vì tốc độ khác nhau. TM[i][j] = DM[i][j] / speed * 60.
- **Tại sao lưu TM riêng?** Truck đi đường lớn (nhanh hơn), bike đi hẻm (chậm hơn). Sau này có thể dùng actual road network time thay vì Euclidean.
- **restricted**: bike-only customer (hẻm nhỏ).
- **tags**: metadata tùy chọn ("heavy", "fragile", v.v.) cho mở rộng.

## NPZ Format (for numpy loading)

```python
# Nhanh hơn JSON cho instance lớn
np.savez("instance.npz",
    customers=customers,      # (N, 6) float64
    restricted=restricted,    # (N,) int8
    depot=depot,              # (2,) float64
    vehicles=vehicles,        # (K, 4) float64
    dist_matrix=dist_matrix,  # (N+1, N+1) float64
    tm_truck=tm_truck,        # (N+1, N+1) float64
    tm_bike=tm_bike,          # (N+1, N+1) float64
)
```

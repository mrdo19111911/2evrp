"""Run solver in background thread, emit progress over WebSocket."""
import asyncio
import time
import threading
import numpy as np

from src.data.io import load_instance
from src.data.distance import compute_dist_matrix
from src.data.constants import (
    CFG_MAX_ITERATIONS, COL_X, COL_Y, COL_DEMAND,
    COL_TW_OPEN, COL_TW_CLOSE, COL_SERVICE, COL_RESTRICTED,
)
from src.alns.loop import solve
from server.solution_serializer import serialize_solution, serialize_eval


def _serialize_instance(customers, depot):
    """Serialize instance data so frontend can show customers on map immediately."""
    custs = []
    for i in range(len(customers)):
        custs.append({
            "id": i,
            "x": float(customers[i, COL_X]),
            "y": float(customers[i, COL_Y]),
            "demand_g": int(customers[i, COL_DEMAND]),
            "tw_open_s": int(customers[i, COL_TW_OPEN]),
            "tw_close_s": int(customers[i, COL_TW_CLOSE]),
            "service_s": int(customers[i, COL_SERVICE]),
            "restricted": bool(customers[i, COL_RESTRICTED]),
        })
    return {
        "type": "instance_loaded",
        "depot": {"x": float(depot[0]), "y": float(depot[1])},
        "customers": custs,
        "n_customers": len(customers),
    }


class SolverRunner:
    """Manages one solver run in a background thread."""

    def __init__(self, broadcast_fn, loop):
        self.broadcast = broadcast_fn
        self.loop = loop
        self.running = False
        self.thread = None
        self.customers = None
        self.depot = None
        self.vehicles = None
        self.dist_matrix = None

    def start(self, instance_path, n_trucks, n_bikes, max_iterations, seed):
        if self.running:
            return {"error": "solver already running"}
        self.running = True
        self.customers, self.depot, self.vehicles = load_instance(instance_path)
        self.dist_matrix = compute_dist_matrix(self.depot, self.customers)

        # Emit instance immediately so frontend shows customers on map
        self._emit(_serialize_instance(self.customers, self.depot))
        self._emit({"type": "status", "state": "loading", "message": "Building initial solution..."})

        self.thread = threading.Thread(
            target=self._run,
            args=(n_trucks, n_bikes, max_iterations, seed),
            daemon=True,
        )
        self.thread.start()
        return {"status": "started"}

    def stop(self):
        self.running = False

    def _emit(self, msg):
        asyncio.run_coroutine_threadsafe(self.broadcast(msg), self.loop)

    def _run(self, n_trucks, n_bikes, max_iterations, seed):
        start_time = time.time()
        last_emit = 0.0
        last_sol_emit = 0.0

        def callback(iteration, best_sol, eval_arr, log_dict):
            nonlocal last_emit, last_sol_emit
            if not self.running:
                raise StopIteration("stopped by user")

            now = time.time()
            if now - last_emit < 0.1:
                return
            last_emit = now

            pct = round(iteration / max_iterations * 100, 1)
            msg = {
                "type": "iteration",
                "iter": int(iteration),
                "max_iter": max_iterations,
                "pct": pct,
                "elapsed_s": round(now - start_time, 2),
                "eval": serialize_eval(eval_arr),
                "log_tail": {
                    "fitness": log_dict["fitness"][-200:] if log_dict["fitness"] else [],
                    "best_fitness": log_dict["best_fitness"][-200:] if log_dict["best_fitness"] else [],
                    "temperature": log_dict["temperature"][-200:] if log_dict["temperature"] else [],
                },
            }

            # Send solution snapshot every 2 seconds so map updates live
            if now - last_sol_emit >= 2.0:
                last_sol_emit = now
                msg["type"] = "solution_update"
                msg["solution"] = serialize_solution(best_sol, self.customers, self.depot)

            self._emit(msg)

        try:
            best_sol, best_fitness, archive, log = solve(
                self.customers, self.depot, self.vehicles, self.dist_matrix,
                n_trucks, n_bikes,
                config_overrides={CFG_MAX_ITERATIONS: max_iterations},
                seed=seed,
                callback=callback,
            )

            solution_data = serialize_solution(best_sol, self.customers, self.depot)
            self._emit({
                "type": "done",
                "solution": solution_data,
                "fitness": int(best_fitness),
                "elapsed_s": round(time.time() - start_time, 2),
                "log": log,
                "archive_size": len(archive),
            })
        except StopIteration:
            self._emit({"type": "stopped"})
        except Exception as e:
            self._emit({"type": "error", "message": str(e)})
        finally:
            self.running = False

"""REST endpoints: start/stop solver, get status."""
import asyncio
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_runner = None


def _get_runner():
    global _runner
    if _runner is None:
        from server.app import broadcast
        # Get the RUNNING event loop (called from async context via FastAPI)
        loop = asyncio.get_running_loop()
        from server.solver_runner import SolverRunner
        _runner = SolverRunner(broadcast, loop)
    return _runner


class SolveRequest(BaseModel):
    instance_path: str = "data/grid_20x20.json"
    n_trucks: int = 5
    n_bikes: int = 15
    max_iterations: int = 50000
    seed: int = 42


@router.post("/solve")
async def start_solve(req: SolveRequest):
    runner = _get_runner()
    return runner.start(req.instance_path, req.n_trucks, req.n_bikes,
                        req.max_iterations, req.seed)


@router.post("/stop")
async def stop_solve():
    runner = _get_runner()
    runner.stop()
    return {"status": "stopping"}


@router.get("/status")
async def get_status():
    runner = _get_runner()
    return {"running": runner.running}


@router.get("/instance")
async def load_instance_data(path: str = "data/grid_20x20.json"):
    """Load instance and return customers + depot for map preview."""
    from server.solver_runner import _serialize_instance
    from src.data.io import load_instance
    customers, depot, vehicles = load_instance(path)
    return _serialize_instance(customers, depot)

"""FastAPI server: REST for config, WebSocket for live solver updates."""
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="2E-VRP Solver Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

clients: list[WebSocket] = []


async def broadcast(msg: dict):
    """Send JSON to all connected WebSocket clients."""
    text = json.dumps(msg)
    disconnected = []
    for ws in clients:
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        clients.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in clients:
            clients.remove(ws)


@app.get("/api/instances")
def list_instances():
    """List available instance JSON files in data/."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    files = sorted(data_dir.glob("*.json"))
    return [{"name": f.name, "path": str(f)} for f in files]


from server.routes import router  # noqa: E402
app.include_router(router, prefix="/api")

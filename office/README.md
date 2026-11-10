# DevTeam Office

The Phaser office replays saved runs and can display a live planning run with human approval.

## Start a live run

From the repository root, start the local FastAPI server:

```powershell
.venv\Scripts\python -m server
```

Open `http://127.0.0.1:8000`, enter a project request, and choose **Start run**. Events stream into
the office as the planning agents work. When the task board is ready, the meeting room appears; edit
its JSON if needed, then approve or reject the plan. The server binds to localhost and writes the
usual event log and checkpoint for the run.

## Open the replay

1. Open `index.html` and choose **Load sample**, or select any run's `events.jsonl` with **Open events.jsonl**.
2. Use Play, Pause, speed, and the timeline slider to inspect the run.

If you only want to view a saved run and the browser blocks the sample fetch when opened as a local
file, serve this folder over localhost:

```powershell
cd office
python -m http.server 8000
```

Then open `http://localhost:8000`. The pinned Phaser 3 script is loaded from cdnjs.

The viewer builds its rooms, desks, and avatars in code. It shows only agents named in the run;
handoffs use grid pathfinding and bubbles display each event's objective or result.

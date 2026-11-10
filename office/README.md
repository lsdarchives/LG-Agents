# DevTeam Office Replay

This static Phaser page replays an existing `runs/<run_id>/events.jsonl` file. It makes no model
calls and can be used with the sample log included here.

## Open the replay

1. Open `index.html` and choose **Load sample**, or select any run's `events.jsonl` with **Open events.jsonl**.
2. Use Play, Pause, speed, and the timeline slider to inspect the run.

If your browser blocks the sample fetch when opened as a local file, serve this folder over
localhost:

```powershell
cd office
python -m http.server 8000
```

Then open `http://localhost:8000`. The pinned Phaser 3 script is loaded from cdnjs.

The viewer builds its rooms, desks, and avatars in code. It shows only agents named in the loaded
event log; handoffs use grid pathfinding and bubbles display each event's objective or result.

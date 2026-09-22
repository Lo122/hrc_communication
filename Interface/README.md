# HRC watch interface demo

This is a FastAPI interface for one Human Step 0 interaction. FastAPI and the
existing HRC communication runtime run in the same process and share the same
event queue.

The R1 permission question supports `H_ACCEPT`, `H_REFUSE`, and `H_DEFER`.

![Current Human Step 0 permission UI](./ui-preview.png)

Install the backend dependencies from the repository root:

```powershell
uv pip install --python .venv\Scripts\python.exe -r Interface\requirements.txt
```

Start the integrated FastAPI and HRC communication runtime:

```powershell
.venv\Scripts\python.exe -B Interface\server.py
```

Run this command instead of `run_communication.py`, because both programs would
otherwise try to bind the same recognition-event UDP port. Recognition events
arrive through the existing configured event transport. For an H0 test without
the recognition process, start with:

```powershell
.venv\Scripts\python.exe -B Interface\server.py --debug-trigger
```

Then open `http://127.0.0.1:8765`. The page polls `GET /api/permission` and only
enables its buttons while a real Human Step 0 task is in `R_WAITING_RESPONSE`.
The selected response is sent to `POST /api/permission` with the current
`task_instance_id`.

After backend acknowledgement, the page emits a browser event named
`hrc-permission-response`. Its detail uses the same permission event names as
the Python system:

```javascript
window.addEventListener("hrc-permission-response", (event) => {
  console.log(event.detail);
});
```

The backend converts the response into the project's core `Event`, puts it in
the shared `HRCSystem.event_queue`, and immediately processes it through
`TaskManager`. Therefore `H_ACCEPT`, `H_REFUSE`, and `H_DEFER` now have the same
business behavior as their CLI and voice equivalents. In particular, accepting
uses the normal Grasshopper task dispatch path.

"""Walks the full panel cycle with fake CLI/ROS/Grasshopper.

Run from the repo root:  python tests/test_workflow_chain.py
It asserts nothing yet -- it prints the conversation and the Grasshopper
messages so you can read the flow and check the wording out loud.
"""
import sys, time
sys.path.insert(0, ".")
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for layer in ["0_core","1_recognition","2_decision_making","3_communication","4_execution"]:
    sys.path.insert(0, layer)

import config, task_catalog
from events import Event, EventType, RobotTaskState
from message_manager import MessageManager
from cmd_parser import CommandParser
from state_machine import StateMachine
from pending_task import PendingTaskPool
from task_manager import TaskManager
from trigger_manager import TriggerManager
from models import RecognitionResult

SAID = []
class FakeCLI:
    def show_message(self, m): SAID.append(("say", m))
    def show_permission_request(self, m): SAID.append(("ASK", m))
class FakeTimer:
    def start_response_timer(self,*a): pass
    def cancel_response_timer(self,*a): pass
    def start_defer_timer(self,*a): pass
    def cancel_defer_timer(self,*a): pass
class FakeGH:
    sent = []
    def dispatch_task(self, task): self.sent.append(self.build_message(task))
    def build_message(self, task):
        m = config.GH_TASK_MESSAGES.get(task.task_key) or {}
        return {"task_key": task.task_key, "suggested_action": m.get("suggested_action","wait")}
class FakeROS:
    def publish_speed(self,*a): pass
    def publish_human_done(self,*a): SAID.append(("ros","human_done"))
    def publish_free_drive(self,v): SAID.append(("ros",f"free_drive={v}"))
    def publish_pause(self): pass
    def publish_resume(self,*a): pass
    def publish_restart(self): pass
    def publish_cancel(self): pass
    def publish_return_home(self): pass
    def get_latest_joint_positions(self): return None
class FakeLog:
    def log_event(self,e): pass
    def log_transition(self,*a,**k): pass
    def log_message(self,*a,**k): pass

gh = FakeGH()
tm = TaskManager(StateMachine(), PendingTaskPool(), FakeTimer(), MessageManager(),
                 FakeCLI(), gh, FakeROS(), FakeLog())

def send(et, **payload):
    tm.handle_event(Event(event_type=et, source="test", payload=payload))

# --- recognition fires once, at the end of panel cutting -------------------
trig = TriggerManager()
r = RecognitionResult(round_id=1, step_id=task_catalog.STEP_PANEL_CUT, progress=0.4,
                      piece_id=4, confidence=0.9, timestamp=time.time())
trig.update(r)
r2 = RecognitionResult(round_id=1, step_id=task_catalog.STEP_PANEL_CUT, progress=0.85,
                       piece_id=4, confidence=0.9, timestamp=time.time())
events = trig.update(r2)
print(f"recognition produced {len(events)} trigger(s): {events[0].payload['task_key']}\n")
tm.handle_event(events[0])

# --- then the whole chain runs on human answers + robot feedback -----------
script = [
    ("say yes",        lambda: send(EventType.H_ACCEPT)),
    ("robot running",  lambda: send(EventType.ROBOT_RUNNING)),
    ("robot success",  lambda: send(EventType.ROBOT_SUCCESS)),   # -> chains to lift_panel
    ("say yes",        lambda: send(EventType.H_ACCEPT)),
    ("robot running",  lambda: send(EventType.ROBOT_RUNNING)),
    ("robot success",  lambda: send(EventType.ROBOT_SUCCESS)),   # -> chains to hold_for_align
    ("(veto window)",  lambda: send(EventType.DEFER_TIMEOUT)),
    ("robot running",  lambda: send(EventType.ROBOT_RUNNING)),
    ("robot success",  lambda: send(EventType.ROBOT_SUCCESS)),   # free_drive mode -> asks
    ("say free drive", lambda: send(EventType.H_FREE_GO)),
    ("say done",       lambda: send(EventType.H_DONE)),          # -> chains to hold_for_screw
    ("(veto window)",  lambda: send(EventType.DEFER_TIMEOUT)),
    ("robot running",  lambda: send(EventType.ROBOT_RUNNING)),
    ("robot success",  lambda: send(EventType.ROBOT_SUCCESS)),   # -> R_HOLDING, does NOT finish
    ("say not yet",    lambda: send(EventType.H_NOT_YET)),
    ("say secured",    lambda: send(EventType.H_SECURED)),       # -> releases, chains to bring_joint
    ("(veto window)",  lambda: send(EventType.DEFER_TIMEOUT)),
    ("robot running",  lambda: send(EventType.ROBOT_RUNNING)),
    ("robot success",  lambda: send(EventType.ROBOT_SUCCESS)),
]
for label, fn in script:
    SAID.clear()
    fn()
    st = tm.active_task.state.name if tm.active_task else "-- idle --"
    key = tm.active_task.task_key if tm.active_task else ""
    print(f"{label:16} | {st:22} {key}")
    for kind, msg in SAID:
        if kind == "ASK":  print(f"                 |   ROBOT ASKS: {msg}")
        elif kind == "say": print(f"                 |   robot says: {msg}")

print("\nGrasshopper received, in order:")
for m in gh.sent:
    print("  ", m)

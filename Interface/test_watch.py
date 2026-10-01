"""Tests for the state-driven watch screens, watch API and simulator.

Run from the repository root:
    uv run python -m unittest Interface.test_watch
"""

from __future__ import annotations

import asyncio
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import HTTPException

from Interface.server import HRCBridge, SimRequest, WatchCommand
import config
from event_queue import EventQueue
from events import Event, EventType, RobotTaskState
from message_manager import MessageManager
from pending_task import PendingTaskPool
from state_machine import StateMachine
from task_manager import TaskManager
from task_tracker import build_task_tracking
from demo_opening import build_demo_opening
from timer_manager import TimerManager
from watch_screens import _STATE_SCREENS, allowed_commands, build_screen

ROOT = Path(__file__).resolve().parents[1]
PULL = config.STEP_NAMES.index("Pull Cables")


class SimSystem:
    """Real TaskManager (with the task database) and ROS/Grasshopper replaced by
    recording fakes."""

    def __init__(self, demo=False):
        self.event_queue = EventQueue()
        self.gh_dispatcher = MagicMock()
        self.ros = MagicMock()
        self.ros.latest_joint_positions = None
        self.ros.latest_gripper_open = None
        self.ros.get_latest_joint_positions = lambda: self.ros.latest_joint_positions
        self.ros.get_latest_gripper_has_object = (
            lambda: None if self.ros.latest_gripper_open is None else not self.ros.latest_gripper_open
        )
        self.pending_pool = PendingTaskPool()
        self.state_machine = StateMachine()
        self.timer = TimerManager(event_callback=self.event_queue.put)
        self.tracker, self.policy = build_task_tracking(ROOT, logger=MagicMock())
        self.task_manager = TaskManager(
            state_machine=self.state_machine,
            pending_pool=self.pending_pool,
            timer_manager=self.timer,
            message_manager=MessageManager(),
            cli=MagicMock(),
            gh_dispatcher=self.gh_dispatcher,
            ros_communication=self.ros,
            logger=MagicMock(),
            task_tracker=self.tracker,
            trigger_policy=self.policy,
            demo=build_demo_opening() if demo else None,
        )

    def process_events(self) -> None:
        while not self.event_queue.empty():
            self.task_manager.handle_event(self.event_queue.get())


class ScreenTests(unittest.TestCase):
    def test_every_offered_action_is_accepted_by_task_manager_states(self):
        machine = StateMachine()
        for state, spec in _STATE_SCREENS.items():
            for task_id in config.GH_STEP_MESSAGES:
                task = MagicMock(state=state, task_id=task_id, updated_at=0.0)
                screen = build_screen(task, [], machine, now=0.0)
                self.assertLessEqual(len(screen.actions), 3, state)
                self.assertLessEqual(
                    sum(a.role == "primary" for a in screen.actions), 1, state
                )
                for action in screen.actions:
                    self.assertLessEqual(len(action.label), 12)

    def test_idle_and_pending_screens(self):
        machine = StateMachine()
        self.assertEqual(build_screen(None, [], machine, 0).screen, "idle")
        pending = [MagicMock(task_id=config.TASK_LIFT_PANEL)]
        screen = build_screen(None, pending, machine, 0)
        self.assertEqual(screen.screen, "pending")
        self.assertEqual(screen.actions[0].command, "H_EXECUTE_PENDING_TASK")

    def test_idle_offers_only_the_robot_tasks_next_in_the_flow(self):
        machine = StateMachine()
        tracker, _ = build_task_tracking(ROOT, logger=MagicMock())

        def requests(**context):
            screen = build_screen(None, [], machine, 0, tracker=tracker, **context)
            return [a.task_name for a in screen.actions if a.command == "H_REQUEST_ROBOT_TASK"]

        self.assertEqual(requests(), ["Pull Cables"])  # the lift waits for the cables
        tracker.on_task_recognized("Pull Cables", 0.5)
        screen = build_screen(None, [], machine, 0, tracker=tracker)
        self.assertEqual(screen.eyebrow, "PULL CABLES")
        # The human pulls them: confirm it, nothing to ask the robot for yet.
        self.assertEqual([(a.command, a.task_name) for a in screen.actions], [("H_TASK_DONE", None)])
        tracker.confirm_done("Pull Cables")
        self.assertEqual(requests(), ["Lift"])
        for name in ("Lift", "Place", "Align"):
            tracker.confirm_done(name)
        self.assertEqual(requests(), [])
        tracker.on_task_recognized("Screw", 0.3)
        self.assertEqual(requests(), ["Bring Connector"])  # the database has the robot bring only the connector
        for action in build_screen(None, [], machine, 0, tracker=tracker).actions:
            self.assertLessEqual(len(action.label), 12)
        self.assertEqual(requests(opening=True), [])  # the opening dialogue leads

    def test_opening_questions(self):
        machine = StateMachine()
        for question in ("start", "continue"):
            screen = build_screen(None, [], machine, 0, question=question)
            self.assertEqual(screen.screen, "ask-" + question)
            self.assertEqual(allowed_commands(None, [], machine, question=question),
                             {("H_ACCEPT", None), ("H_REFUSE", None)})
            for action in screen.actions:
                self.assertLessEqual(len(action.label), 12)

    def test_no_to_moving_away_reads_not_yet(self):
        machine = StateMachine()
        for task_id, label in ((config.TASK_LEAVE, "Not yet"), (config.TASK_LEAVE_HANDOVER, "Not yet"),
                               (config.TASK_PULL_CABLES, "I'll do it")):
            task = MagicMock(state=RobotTaskState.R_WAITING_RESPONSE, task_id=task_id, updated_at=0.0)
            labels = {a.command: a.label for a in build_screen(task, [], machine, 0).actions}
            self.assertEqual(labels["H_REFUSE"], label)

    def test_next_task_asked_while_the_robot_runs(self):
        machine = StateMachine()
        task = MagicMock(state=RobotTaskState.R_EXECUTING, task_id=config.TASK_PULL_CABLES, updated_at=0.0)
        advance = MagicMock(task_id=config.TASK_LIFT_PANEL)
        screen = build_screen(task, [], machine, 0, advance=advance)
        self.assertEqual(screen.screen, "ask-next")
        self.assertEqual(screen.title, "Lift the panel?")
        self.assertEqual([a.command for a in screen.actions], ["H_ACCEPT", "H_REFUSE", "H_CANCEL"])
        self.assertTrue(screen.actions[-1].hold)

        # Answered: the running screen is back, and says what follows.
        screen = build_screen(task, [], machine, 0, advance=advance, advance_answer="H_ACCEPT")
        self.assertEqual(screen.screen, "running")
        self.assertEqual(screen.detail, "Next: lift the panel")


class WatchOpeningTests(unittest.IsolatedAsyncioTestCase):
    """--demo: the watch carries the opening dialogue."""

    async def asyncSetUp(self):
        self.system = SimSystem(demo=True)
        self.bridge = HRCBridge(self.system, simulate=True, sim_task_seconds=0.01)
        self.system.event_queue.put(Event(EventType.DEMO_START, "test"))
        self.system.process_events()

    async def asyncTearDown(self):
        self.system.timer.cancel_response_timer()
        for name in list(self.system.timer.timers):
            self.system.timer.cancel(name)

    async def cmd(self, command):
        return await self.bridge.submit_watch_command(WatchCommand(command=command))

    async def test_start_then_the_human_pulls_then_moves_on(self):
        snap = await self.bridge.watch_state()
        self.assertEqual(snap["screen"]["screen"], "ask-start")
        snap = await self.cmd("H_ACCEPT")
        self.assertEqual((snap["task"]["task_id"], snap["screen"]["screen"]),
                         (config.TASK_PULL_CABLES, "ask"))
        snap = await self.cmd("H_REFUSE")              # the human pulls them
        self.assertEqual(snap["screen"]["screen"], "ask-continue")
        self.assertEqual(snap["pending"], [])
        snap = await self.cmd("H_ACCEPT")              # move on
        self.assertEqual((snap["task"]["task_id"], snap["screen"]["screen"]),
                         (config.TASK_LIFT_PANEL, "ask"))

    async def test_no_robot_requests_while_the_opening_waits(self):
        snap = await self.cmd("H_REFUSE")              # not yet
        self.assertEqual(snap["screen"]["screen"], "idle")
        self.assertEqual(snap["screen"]["actions"], [])


class WatchFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.system = SimSystem()
        self.bridge = HRCBridge(self.system, simulate=True, sim_task_seconds=0.01)
        self.bridge.sim.start_delay = 0.0

    async def asyncTearDown(self):
        self.system.timer.cancel_response_timer()
        self.system.timer.cancel_defer_timer()

    async def step(self, n=3):
        for i in range(n):
            self.bridge.sim.tick(time.time() + 1.0 + i)
            self.system.process_events()

    async def cmd(self, command):
        return await self.bridge.submit_watch_command(WatchCommand(command=command))

    async def request(self, task_name):
        return await self.bridge.submit_watch_command(
            WatchCommand(command="H_REQUEST_ROBOT_TASK", task_name=task_name))

    async def test_full_lift_panel_flow(self):
        await self.bridge.simulate(SimRequest(event="RECOGNITION_TRIGGER", step_id=PULL))
        snap = await self.bridge.watch_state()
        self.assertEqual(snap["task"]["task_id"], config.TASK_LIFT_PANEL)
        self.assertEqual(snap["screen"]["screen"], "ask")
        self.assertIsNotNone(snap["screen"]["countdown_deadline"])

        await self.cmd("H_ACCEPT")
        await self.step()
        if config.LIFT_ASKS_FREE_DRIVE:
            snap = await self.bridge.watch_state()
            self.assertEqual(snap["state"], "R_WAITING_FREE_DRIVE")
            await self.cmd("H_FREE_GO")
        snap = await self.bridge.watch_state()
        self.assertEqual(snap["screen"]["screen"], "free-drive")

        await self.cmd("H_DONE")
        snap = await self.cmd("H_SCREW_DONE")
        self.assertEqual(snap["task"]["task_id"], config.TASK_LEAVE)
        self.assertEqual(snap["screen"]["screen"], "ask")

    async def test_connector_hand_over_not_yet_then_give_me(self):
        # Asked for by voice ("bring the connector"); the watch answers from then on.
        self.system.event_queue.put(Event(EventType.H_REQUEST_ROBOT_TASK, "test",
                                          payload={"task_name": "Bring Connector"}))
        self.system.process_events()
        await self.cmd("H_ACCEPT")
        await self.step()
        snap = await self.bridge.watch_state()
        self.assertEqual(snap["screen"]["screen"], "ask-handover")

        snap = await self.cmd("H_REFUSE")
        self.assertEqual(snap["screen"]["screen"], "holding-handover")
        self.system.ros.publish_gripper_open.assert_not_called()

        snap = await self.cmd("H_HANDOVER")
        self.system.ros.publish_gripper_open.assert_called_once()
        self.assertEqual(snap["task"]["task_id"], config.TASK_LEAVE_HANDOVER)
        self.assertEqual(snap["screen"]["screen"], "defer")
        self.assertEqual(snap["screen"]["detail"], "Stepping back")
        self.assertEqual(snap["screen"]["countdown_total"],
                         config.HANDOVER_LEAVE_DELAY_S[config.TASK_BRING_CONNECTOR])

    async def test_idle_request_and_task_done(self):
        snap = await self.bridge.watch_state()
        self.assertEqual(snap["screen"]["screen"], "idle")
        requests = [a["task_name"] for a in snap["screen"]["actions"]
                    if a["command"] == "H_REQUEST_ROBOT_TASK"]
        self.assertEqual(requests, ["Pull Cables"])
        for not_yet in ("Lift", "Bring back Tool"):     # not on the screen
            with self.assertRaises(HTTPException) as raised:
                await self.request(not_yet)
            self.assertEqual(raised.exception.status_code, 409)

        snap = await self.request("Pull Cables")
        self.assertEqual(snap["task"]["task_id"], config.TASK_PULL_CABLES)
        self.assertEqual(snap["screen"]["screen"], "ask")

        snap = await self.cmd("H_DEFER")              # later: to the pending pool
        self.assertEqual(snap["screen"]["screen"], "pending")

    async def test_no_hands_the_task_to_the_human(self):
        await self.request("Pull Cables")
        snap = await self.cmd("H_REFUSE")             # I'll do it
        self.assertEqual(snap["screen"]["screen"], "idle")
        self.assertEqual(snap["pending"], [])

    async def test_rejects_command_not_on_screen(self):
        await self.bridge.simulate(SimRequest(event="RECOGNITION_TRIGGER", step_id=PULL))
        with self.assertRaises(HTTPException) as raised:
            await self.cmd("H_PAUSE")
        self.assertEqual(raised.exception.status_code, 409)
        with self.assertRaises(HTTPException) as raised:
            await self.cmd("ROBOT_SUCCESS")
        self.assertEqual(raised.exception.status_code, 422)

    async def test_pause_speed_and_stop_to_home(self):
        await self.bridge.simulate(SimRequest(event="RECOGNITION_TRIGGER", step_id=PULL))
        await self.cmd("H_ACCEPT")
        self.bridge.sim.auto_robot = False
        self.system.event_queue.put(Event(EventType.ROBOT_RUNNING, "test"))
        self.system.process_events()
        snap = await self.cmd("H_SPEEDUP")
        self.assertAlmostEqual(snap["speed"]["value"], config.DEFAULT_SPEED + config.SPEED_STEP)
        snap = await self.cmd("H_PAUSE")
        self.assertEqual(snap["screen"]["screen"], "paused")
        snap = await self.cmd("H_CANCEL")
        self.assertEqual(snap["screen"]["screen"], "ask-home")
        self.bridge.sim.auto_robot = True
        await self.cmd("H_RETURN_HOME")
        self.bridge.sim._homed_at = time.time()
        await self.step()
        snap = await self.bridge.watch_state()
        self.assertEqual(snap["screen"]["screen"], "idle")

    async def test_deferred_task_can_be_started_from_pending(self):
        await self.bridge.simulate(SimRequest(event="RECOGNITION_TRIGGER", step_id=PULL))
        snap = await self.cmd("H_DEFER")  # later: pending (a no hands it to the human)
        self.assertEqual(snap["screen"]["screen"], "pending")
        snap = await self.cmd("H_EXECUTE_PENDING_TASK")
        # Offered again: it only starts after a fresh yes.
        self.assertEqual(snap["state"], "R_WAITING_RESPONSE")

    async def test_unknown_step_is_rejected(self):
        with self.assertRaises(HTTPException) as raised:
            await self.bridge.simulate(SimRequest(event="RECOGNITION_TRIGGER",
                                                  step_id=len(config.STEP_NAMES)))
        self.assertEqual(raised.exception.status_code, 422)


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    """The watch must show "speaking" while the robot talks, not after it."""

    async def test_watch_answers_while_the_robot_is_speaking(self):
        system = SimSystem()
        spoken = []

        class SlowTTS:                       # pyttsx3 blocks exactly like this
            def speak(self, text, *args, **kwargs):
                time.sleep(1.0)
                spoken.append(text)

        system.tts = SlowTTS()
        system.communication = SimpleNamespace(
            mode=None, show_message=lambda message, **kwargs: None
        )
        bridge = HRCBridge(system, simulate=True)

        system.tts.speak("May I lift the panel?")
        started = time.time()
        snapshot = await bridge.watch_state()
        answered_in = time.time() - started

        self.assertLess(answered_in, 0.2, "the watch waited for the sentence to end")
        self.assertTrue(snapshot["voice"]["speaking"])
        time.sleep(1.4)
        self.assertEqual(spoken, ["May I lift the panel?"])


class ResponsivenessTests(unittest.IsolatedAsyncioTestCase):
    """A slow runtime tick must not stop the watch from being answered.

    The voice layer joins the microphone thread for up to a second whenever a
    question is announced; run inside the event loop that froze the screen.
    """

    async def test_watch_is_answered_during_a_slow_tick(self):
        class SlowSystem(SimSystem):
            def close(self):
                pass

            def process_events(self):
                time.sleep(0.6)          # like voice.stop_listening()'s join
                super().process_events()

        bridge = HRCBridge(SlowSystem(), simulate=True)
        await bridge.start()
        try:
            await asyncio.sleep(1.0)     # let one slow tick be under way
            worst = 0.0
            for _ in range(10):
                started = time.time()
                await bridge.watch_state()
                worst = max(worst, time.time() - started)
                await asyncio.sleep(0.05)
        finally:
            await bridge.stop()
        self.assertLess(worst, 0.1, "the watch waited for the slow tick")


class MicrophoneTests(unittest.TestCase):
    """The robot must not listen to itself while it speaks.

    Vosk decoding audio while SAPI is still talking shares the sound device and
    the CPU, and the spoken sentence comes out chopped.
    """

    def test_listening_waits_until_the_sentence_is_over(self):
        opened = []
        system = SimSystem()
        system.tts = SimpleNamespace(speak=lambda text, *a, **k: None)
        system.voice = SimpleNamespace(
            start_listening=lambda *a, **k: opened.append(time.time())
        )
        system.communication = SimpleNamespace(
            mode=None, show_message=lambda message, **kwargs: None
        )
        bridge = HRCBridge(system, simulate=True)

        system.tts.speak("May I lift the panel?")
        system.voice.start_listening(lambda *_: None, lambda *_: None)
        self.assertEqual(opened, [], "the microphone opened during the sentence")

        deadline = bridge._speaking_until + 0.6
        while time.time() < deadline and not opened:
            time.sleep(0.05)
        self.assertTrue(opened, "the microphone never opened after the sentence")


if __name__ == "__main__":
    unittest.main()

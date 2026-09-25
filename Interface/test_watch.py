"""Tests for the state-driven watch screens, watch API and simulator.

Run from the repository root:
    uv run python -m unittest Interface.test_watch
"""

from __future__ import annotations

import asyncio
import time
import unittest
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
from timer_manager import TimerManager
from watch_screens import _STATE_SCREENS, build_screen


class SimSystem:
    """Real TaskManager with ROS/Grasshopper replaced by recording fakes."""

    def __init__(self):
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
        self.task_manager = TaskManager(
            state_machine=self.state_machine,
            pending_pool=self.pending_pool,
            timer_manager=self.timer,
            message_manager=MessageManager(),
            cli=MagicMock(),
            gh_dispatcher=self.gh_dispatcher,
            ros_communication=self.ros,
            logger=MagicMock(),
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

    async def test_full_lift_panel_flow(self):
        await self.bridge.simulate(SimRequest(event="RECOGNITION_TRIGGER", step_id=0))
        snap = await self.bridge.watch_state()
        self.assertEqual(snap["screen"]["screen"], "ask")
        self.assertIsNotNone(snap["screen"]["countdown_deadline"])

        await self.cmd("H_ACCEPT")
        await self.step()
        snap = await self.bridge.watch_state()
        self.assertEqual(snap["state"], "R_WAITING_FREE_DRIVE")

        await self.cmd("H_FREE_GO")
        await self.cmd("H_DONE")
        snap = await self.cmd("H_SCREW_DONE")
        self.assertEqual(snap["task"]["task_id"], config.TASK_LEAVE)
        self.assertEqual(snap["screen"]["screen"], "ask")

    async def test_rejects_command_not_on_screen(self):
        await self.bridge.simulate(SimRequest(event="RECOGNITION_TRIGGER", step_id=0))
        with self.assertRaises(HTTPException) as raised:
            await self.cmd("H_PAUSE")
        self.assertEqual(raised.exception.status_code, 409)
        with self.assertRaises(HTTPException) as raised:
            await self.cmd("ROBOT_SUCCESS")
        self.assertEqual(raised.exception.status_code, 422)

    async def test_pause_speed_and_stop_to_home(self):
        await self.bridge.simulate(SimRequest(event="RECOGNITION_TRIGGER", step_id=4))
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

    async def test_refused_task_can_be_started_from_pending(self):
        await self.bridge.simulate(SimRequest(event="RECOGNITION_TRIGGER", step_id=5))
        snap = await self.cmd("H_REFUSE")
        self.assertEqual(snap["screen"]["screen"], "pending")
        snap = await self.cmd("H_EXECUTE_PENDING_TASK")
        self.assertEqual(snap["state"], "R_ACCEPTED")


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

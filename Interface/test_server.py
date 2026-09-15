"""Focused tests for the HTTP-to-HRC permission bridge."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace

from fastapi import HTTPException

from Interface.server import HRCBridge, PermissionRequest
from event_queue import EventQueue
from events import Event, EventType, RobotTaskState
from message_manager import MessageManager
from pending_task import PendingTaskPool
from state_machine import StateMachine
from task_manager import TaskManager
from timer_manager import TimerManager


class RecordingQueue:
    def __init__(self):
        self.events = []

    def put(self, event: Event) -> None:
        self.events.append(event)


class RecordingSystem:
    def __init__(self, task):
        self.task_manager = SimpleNamespace(active_task=task)
        self.event_queue = RecordingQueue()
        self.processed_events = []

    def process_events(self) -> None:
        self.processed_events.extend(self.event_queue.events)
        self.event_queue.events.clear()


class BusinessSystem:
    """Small real TaskManager harness without ROS or Grasshopper I/O."""

    def __init__(self):
        self.event_queue = EventQueue()
        self.gh_dispatcher = MagicMock()
        self.ros = MagicMock()
        self.timer = TimerManager(event_callback=self.event_queue.put)
        self.task_manager = TaskManager(
            state_machine=StateMachine(),
            pending_pool=PendingTaskPool(),
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


def h0_task(task_instance_id: str = "round_0_task_1_piece_1"):
    return SimpleNamespace(
        state=RobotTaskState.R_WAITING_RESPONSE,
        step_id=0,
        task_id=1,
        task_instance_id=task_instance_id,
    )


class HRCBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_accept_reaches_real_task_manager_business_logic(self):
        system = BusinessSystem()
        system.event_queue.put(
            Event(
                event_type=EventType.RECOGNITION_TRIGGER,
                source="test_recognition",
                payload={
                    "step_id": 0,
                    "piece_id": 1,
                    "round_id": 0,
                    "progress": 1.0,
                },
            )
        )
        system.process_events()
        task = system.task_manager.active_task
        bridge = HRCBridge(system)

        await bridge.submit_permission(
            PermissionRequest(
                command="H_ACCEPT",
                task_instance_id=task.task_instance_id,
            )
        )

        self.assertIs(task.state, RobotTaskState.R_ACCEPTED)
        system.gh_dispatcher.dispatch_task.assert_called_once_with(task)

    async def test_permission_uses_core_event_and_shared_queue(self):
        system = RecordingSystem(h0_task())
        bridge = HRCBridge(system)

        result = await bridge.submit_permission(
            PermissionRequest(
                command="H_ACCEPT",
                task_instance_id="round_0_task_1_piece_1",
            )
        )

        self.assertEqual(result.event_type, "H_ACCEPT")
        self.assertEqual(len(system.processed_events), 1)
        event = system.processed_events[0]
        self.assertIsInstance(event, Event)
        self.assertIs(event.event_type, EventType.H_ACCEPT)
        self.assertEqual(event.source, "human_watch")
        self.assertEqual(event.task_instance_id, "round_0_task_1_piece_1")

    async def test_stale_task_instance_is_rejected(self):
        system = RecordingSystem(h0_task())
        bridge = HRCBridge(system)

        with self.assertRaises(HTTPException) as raised:
            await bridge.submit_permission(
                PermissionRequest(command="H_REFUSE", task_instance_id="old_task")
            )

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(system.processed_events, [])

    async def test_only_waiting_h0_task_is_exposed(self):
        task = h0_task()
        task.state = RobotTaskState.R_EXECUTING
        bridge = HRCBridge(RecordingSystem(task))

        permission = await bridge.permission_state()

        self.assertFalse(permission.available)


if __name__ == "__main__":
    unittest.main()

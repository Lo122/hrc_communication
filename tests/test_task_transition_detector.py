"""Task detectors and how their signals reach the task state; offline."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for layer in ("0_core", "1_recognition", "2_decision_making", "3_communication", "4_execution"):
    sys.path.insert(0, str(ROOT / layer))

import config
from event_queue import EventQueue
from events import Event, EventType as E, RobotTaskState as S, TaskStatus as T
from force_monitors import ScrewingThresholds
from gh_dispatcher import GHDispatcher
from cmd_parser import CommandParser
from message_manager import MessageManager
from pending_task import PendingTaskPool
from ros_communication import ROSCommunication
from state_machine import StateMachine
from task_manager import TaskManager
from task_tracker import build_task_tracking
from sequence_step_selector import StepChoice
from task_transition_detector import (DONE_SIGNAL, PANEL_SECURED, PROGRESS_SIGNAL, SCREW_COUNT, TCP_WEIGHT_CHANGE,
                                      Detector, DetectorContext, DurationDoneDetector, ForceScrewDetector,
                                      ProgressDoneDetector, ScrewCountDetector, Signal,
                                      TaskTransitionDetectors, build_detectors)

T0 = 1000.0
HZ = 125.0
PUSHES = (4.0, 8.0, 12.0, 16.0)  # 2.5 s each; 1.5 s pauses, shorter than quiet_s (2 s)
SHIFT_AT = 19.5                  # the panel's weight going over to the frame: +6 N
HOLDING = SimpleNamespace(task_instance_id="hold-1", state=S.R_HOLDING)
CLIMB = [round(0.05 * i, 2) for i in range(15)]  # 0.0 .. 0.7 in 3 s, like one real screw


def wrench_at(s: float) -> list[float]:
    fz = -25.0
    if any(start <= s < start + 2.5 for start in PUSHES):
        fz -= 12.0
    if s >= SHIFT_AT:
        fz += 6.0
    return [0.0, 0.0, fz, 0.0, 0.0, 0.0]


def screwing(progress_per_screw, then="Connect Cables", step_s=0.2):
    """Recognition's (t, task, progress) stream: one progress run per screw, then the
    next task."""
    samples, t = [], T0
    for run in progress_per_screw:
        for progress in run:
            samples.append((t, "Screw", progress))
            t += step_s
    samples.append((t, then, 0.1))
    return samples


class ScrewCountDetectorTests(unittest.TestCase):
    def setUp(self):
        self.tracker, _policy = build_task_tracking(ROOT, logger=Mock())
        self.detector = ScrewCountDetector(screws_per_panel=6, high=0.5, low=0.15, min_interval_s=1.0)

    def feed(self, samples, held_piece_id=1, chunk=5):
        """Hand the stream over a few samples per pass, as the main loop does."""
        seen = []
        for i in range(0, len(samples), chunk):
            part = samples[i:i + chunk]
            context = DetectorContext(part[-1][0], HOLDING if held_piece_id else None, held_piece_id,
                                      self.tracker, recognition=part)
            seen += self.detector.update(context)
        return seen

    def counted(self, seen):
        return [(s.piece_id, round(s.value * 6)) for s in seen if s.name == SCREW_COUNT]

    def test_each_climb_and_fall_is_one_screw_and_all_six_is_done(self):
        seen = self.feed(screwing([CLIMB] * 6))
        self.assertEqual(self.counted(seen), [(1, n) for n in range(1, 7)])
        self.assertEqual([s.value for s in seen if s.name == PROGRESS_SIGNAL],
                         [s.value for s in seen if s.name == SCREW_COUNT])
        self.assertEqual(seen[-1].name, PANEL_SECURED)
        self.assertEqual(sum(s.name == PANEL_SECURED for s in seen), 1)
        self.assertNotIn(DONE_SIGNAL, [s.name for s in seen])  # Screw itself is not done
        # The first five drop into the next screw; the last ends when recognition moves on.
        whys = [s.details["why"] for s in seen if s.name == SCREW_COUNT]
        self.assertEqual(whys, ["progress fell back"] * 5 + ["recognition moved on to Connect Cables"])
        self.assertTrue(all((s.task_name, s.source) == ("Screw", "screw count") for s in seen))

    def test_jitter_between_the_thresholds_counts_nothing(self):
        wobble = [0.3, 0.45, 0.35, 0.48, 0.2, 0.4, 0.3]      # never reaches high
        after = [0.6, 0.7, 0.4, 0.55, 0.1, 0.3, 0.12, 0.35]   # one screw: 0.1 after 0.7
        self.assertEqual(self.counted(self.feed(screwing([wobble, after], then="Screw"))), [(1, 1)])

    def test_counts_too_close_together_are_one(self):
        self.assertEqual(self.counted(self.feed(screwing([[0.0, 0.6, 0.1, 0.6, 0.1]], then="Screw"))),
                         [(1, 1)])

    def test_starts_over_for_a_new_panel_and_stops_once_screw_is_done(self):
        self.feed(screwing([CLIMB] * 2, then="Screw"))
        self.tracker.confirm_done("Screw", piece_id=1)
        self.assertEqual(self.feed(screwing([CLIMB] * 2)), [])
        # Piece 2 counts from 1 again.
        self.assertEqual(self.counted(self.feed(screwing([CLIMB] * 2, then="Screw"), held_piece_id=2)),
                         [(2, 1), (2, 2)])

    def test_counts_on_the_humans_piece_without_a_held_panel(self):
        self.assertEqual(self.counted(self.feed(screwing([CLIMB] * 2), held_piece_id=None)),
                         [(1, 1), (1, 2)])


class ForceScrewDetectorTests(unittest.TestCase):
    def setUp(self):
        self.tracker, _policy = build_task_tracking(ROOT, logger=Mock())
        self.hold = HOLDING
        self.detector = self.make_detector(min_progress=0.5)
        self.clock = 0.0

    def make_detector(self, min_progress):
        return ForceScrewDetector(ScrewingThresholds(), min_progress=min_progress, weight_change_n=5.0)

    def run_until(self, end_s, active_task=None, tracker=True, screws=3, wrench=wrench_at):
        """Feed the synthetic wrench in 0.1 s passes, with `screws` of 6 counted so far;
        returns (seconds, signal) pairs."""
        seen = []
        active_task = active_task or self.hold
        reported = {("Screw", 1, SCREW_COUNT): screws / 6}
        while self.clock < end_s:
            start, self.clock = self.clock, round(self.clock + 0.1, 6)
            samples = [(T0 + i / HZ, wrench(i / HZ))
                       for i in range(int(start * HZ), int(self.clock * HZ))]
            context = DetectorContext(T0 + start, active_task, 1, self.tracker if tracker else None,
                                      samples, reported=reported)
            seen += [(self.clock, signal) for signal in self.detector.update(context)]
        return seen

    def test_reports_weight_change_then_done_once(self):
        seen = self.run_until(35.0)
        self.assertEqual([(signal.name, signal.value) for _s, signal in seen],
                         [(TCP_WEIGHT_CHANGE, True), (PANEL_SECURED, True)])
        self.assertTrue(all((s.task_name, s.piece_id, s.source) == ("Screw", 1, "force screw")
                            for _t, s in seen))
        # Loud from the shift until max_push_s (8 s) calls it a level, then quiet_s (2 s).
        self.assertAlmostEqual(seen[-1][0], SHIFT_AT + 8.0 + 2.0, delta=0.3)

    def test_done_waits_until_enough_screws_are_counted(self):
        self.assertNotIn(PANEL_SECURED, [s.name for _t, s in self.run_until(35.0, screws=2)])
        self.assertEqual([s.name for _t, s in self.run_until(35.1, screws=3)], [PANEL_SECURED])
        self.assertEqual(self.run_until(36.0), [])

    def test_no_done_signal_for_a_screw_already_done(self):
        self.tracker.confirm_done("Screw", piece_id=1)
        self.assertNotIn(PANEL_SECURED, [s.name for _t, s in self.run_until(35.0)])

    def test_force_alone_without_progress_gate(self):
        self.detector = self.make_detector(min_progress=None)
        self.assertIn(PANEL_SECURED, [s.name for _t, s in self.run_until(35.0, tracker=False, screws=0)])

    def weight_detector(self):
        return ForceScrewDetector(ScrewingThresholds(), min_progress=0.5, weight_change_n=5.0,
                                  weight_range_n=(5.0, 40.0), weight_steady_s=10.0)

    def done(self, seen):
        return [(s, signal) for s, signal in seen if signal.name == PANEL_SECURED]

    def test_steady_weight_on_the_frame_is_done_without_counted_screws(self):
        self.detector = self.weight_detector()
        done = self.done(self.run_until(40.0, screws=0))  # recognition counted nothing
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0][1].details["because"], "panel weight steady on the frame")
        # The +6 N from SHIFT_AT on is in range and no push: 10 s later.
        self.assertAlmostEqual(done[0][0], SHIFT_AT + 10.0, delta=0.3)

    def test_leaving_the_weight_range_starts_the_wait_over(self):
        def dips(s):
            wrench = wrench_at(s)
            if 25.0 <= s < 26.0:
                wrench[2] = -25.0  # back to the hold's own load for a second
            return wrench

        self.detector = self.weight_detector()
        done = self.done(self.run_until(45.0, screws=0, wrench=dips))
        self.assertAlmostEqual(done[0][0], 26.0 + 10.0, delta=0.3)

    def test_load_outside_the_weight_range_never_counts(self):
        def heavy(s):
            return [0.0, 0.0, -25.0 + (60.0 if s >= SHIFT_AT else 0.0), 0.0, 0.0, 0.0]

        self.detector = self.weight_detector()
        seen = self.run_until(45.0, screws=0, wrench=heavy)
        self.assertEqual(self.done(seen), [])
        self.assertIn(TCP_WEIGHT_CHANGE, [signal.name for _s, signal in seen])

    def test_the_measured_weight_transfer_is_inside_the_configured_range(self):
        # Run 2026-09-28 13:53: the panel's weight going over to the frame changed the
        # load by 45 N. With the configured range that asks to leave soon after.
        def transfer(s):
            return [0.0, 0.0, -25.0 + (45.0 if s >= SHIFT_AT else 0.0), 0.0, 0.0, 0.0]

        self.detector = ForceScrewDetector(ScrewingThresholds(), min_progress=0.5,
                                           weight_change_n=config.TCP_WEIGHT_CHANGE_N,
                                           weight_range_n=config.TCP_WEIGHT_RANGE_N,
                                           weight_steady_s=config.TCP_WEIGHT_STEADY_S)
        done = self.done(self.run_until(45.0, screws=0, wrench=transfer))
        self.assertEqual(len(done), 1)
        # Loud until max_push_s (8 s) calls it a level, then steady for TCP_WEIGHT_STEADY_S.
        self.assertAlmostEqual(done[0][0], SHIFT_AT + 8.0 + config.TCP_WEIGHT_STEADY_S, delta=0.3)

    def test_listens_only_while_holding_and_restarts_per_hold(self):
        lifting = SimpleNamespace(task_instance_id="hold-1", state=S.R_EXECUTING)
        self.run_until(10.0, active_task=lifting)
        self.assertEqual(self.detector.status(), {"listening": False})
        self.clock = 0.0
        self.run_until(7.0)
        first = self.detector.status()["pushes"]
        self.hold = SimpleNamespace(task_instance_id="hold-2", state=S.R_HOLDING)
        self.clock = 0.0
        self.run_until(7.0)
        self.assertEqual((first, self.detector.status()["pushes"]), (1, 1))


class ScrewingMonitorTests(unittest.TestCase):
    def test_wrench_that_starts_late_still_gets_a_full_baseline(self):
        from force_monitors import ScrewingMonitor

        monitor = ScrewingMonitor(ScrewingThresholds())
        monitor.reset(T0)  # the hold starts; the force reader only 30 s later
        for i in range(int(2 * HZ)):
            monitor.update(T0 + 30 + i / HZ, [0.0, 0.0, -25.0 + (i % 2) * 0.4, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(monitor.baseline_at - (T0 + 30), 1.0, delta=0.02)
        self.assertAlmostEqual(monitor.baseline[2], -24.8, delta=0.01)  # averaged, not one sample


class DetectorRunnerTests(unittest.TestCase):
    def test_a_failing_detector_does_not_stop_the_others(self):
        class Broken(Detector):
            name = "broken"

            def update(self, context):
                raise RuntimeError("sensor unplugged")

        class Steady(Detector):
            name = "steady"

            def update(self, context):
                return [self.signal("Screw", 1, SCREW_COUNT, 0.5)]

        runner = TaskTransitionDetectors([Broken(), Steady()])
        context = DetectorContext(T0, None, None, None)
        with self.assertLogs("task_transition_detector", "ERROR"):
            signals = runner.update(context)
        self.assertEqual([(s.source, s.value) for s in signals], [("steady", 0.5)])

    def test_a_later_detector_sees_what_an_earlier_one_reported(self):
        class Counter(Detector):
            name = "counter"

            def update(self, context):
                return [self.signal("Screw", 1, SCREW_COUNT, 0.5)]

        class Reader(Detector):
            name = "reader"
            seen = None

            def update(self, context):
                Reader.seen = context.reported.get(("Screw", 1, SCREW_COUNT))
                return []

        TaskTransitionDetectors([Counter(), Reader()]).update(DetectorContext(T0, None, None, None))
        self.assertEqual(Reader.seen, 0.5)


class TaskSignalHandlingTests(unittest.TestCase):
    def setUp(self):
        self.parser = CommandParser()
        self.output, self.udp = Mock(), Mock()
        self.tracker, self.policy = build_task_tracking(ROOT, logger=Mock())
        self.manager = TaskManager(
            StateMachine(), PendingTaskPool(), Mock(), MessageManager(),
            self.output, GHDispatcher(self.udp), Mock(), Mock(),
            task_tracker=self.tracker, trigger_policy=self.policy,
        )

    def reply(self, text):
        self.manager.handle_event(self.parser.parse(text))

    def signal(self, name, value=True, task_name="Screw", piece_id=1):
        self.manager.handle_event(Event(E.TASK_SIGNAL, "detector:test", payload={
            "task_name": task_name, "piece_id": piece_id, "signal": name, "value": value}))

    def hold(self):
        self.manager.handle_event(Event(E.HUMAN_TASK_UPDATE, "test",
                                        payload={"step_id": config.STEP_NAMES.index("Pull Cables"),
                                                 "round_id": 7, "progress": 0.8}))
        self.reply("yes")
        self.manager.handle_event(Event(E.ROBOT_RUNNING, "test"))
        self.manager.handle_event(Event(E.ROBOT_SUCCESS, "test"))
        self.reply("adjustment done")  # free drive came on at arrival
        self.assertEqual(self.manager.active_task.state, S.R_HOLDING)

    def messages(self):
        return [call.args[0] for call in self.output.show_message.call_args_list]

    def test_panel_secured_while_holding_asks_to_leave_but_leaves_screw_open(self):
        self.hold()
        self.signal(PANEL_SECURED)
        leave = self.manager.active_task
        self.assertEqual((leave.task_id, leave.piece_id, leave.state), (config.TASK_LEAVE, 1, S.R_WAITING_RESPONSE))
        self.assertEqual(self.tracker.get("Screw", 1).status, T.WORKING)  # recognition or "screw done" decides
        self.assertIn("The panel looks secured.", self.messages())
        said = len(self.messages())
        self.signal(PANEL_SECURED)  # a late second one, from the other detector: silent
        self.assertEqual(len(self.messages()), said)
        self.reply("screw done")  # confirms that panel's Screw; the leave question stands
        self.assertEqual(self.tracker.get("Screw", 1).status, T.DONE)
        self.assertIs(self.manager.active_task, leave)
        self.assertEqual(self.output.show_permission_request.call_count, 2)  # the lift, the leave

    def test_panel_secured_without_held_panel_is_only_stored(self):
        self.signal(PANEL_SECURED)
        screw = self.tracker.get("Screw", 1)
        self.assertNotEqual(screw.status, T.DONE)
        self.assertEqual(screw.signals, {PANEL_SECURED: True})
        self.assertIsNone(self.manager.active_task)

    def test_screw_count_unlocks_bring_connector_while_still_holding(self):
        self.hold()
        self.manager.handle_event(Event(E.HUMAN_TASK_UPDATE, "test", payload={
            "step_id": config.STEP_NAMES.index("Screw"), "round_id": 7, "progress": 0.6}))
        self.signal(SCREW_COUNT, 0.25)
        self.assertEqual(list(self.manager.waiting_triggers), [])
        self.signal(SCREW_COUNT, 0.5)
        self.assertEqual(self.tracker.get("Screw", 1).signals, {SCREW_COUNT: 0.5})
        self.assertEqual([(e["task_name"], e["piece_id"]) for e in self.manager.waiting_triggers],
                         [("Bring Connector", 1)])
        self.assertEqual(self.manager.active_task.state, S.R_HOLDING)

    def test_progress_signal_moves_screw_while_holding(self):
        self.hold()
        self.signal(PROGRESS_SIGNAL, 0.5)
        self.assertEqual(self.tracker.get("Screw", 1).progress, 0.5)
        self.assertEqual((self.tracker.reference_task, self.tracker.reference_progress), ("Screw", 0.5))
        self.assertEqual(self.tracker.get("Screw", 1).signals, {})  # progress is not a rule signal

    def test_done_signal_without_held_panel_confirms_the_task(self):
        self.signal(DONE_SIGNAL)
        self.assertEqual((self.tracker.get("Screw", 1).status, self.tracker.get("Screw", 1).executor),
                         (T.DONE, "Human"))
        self.assertEqual(self.manager.active_task.task_id, config.TASK_BRING_CLAMPING_TOOL)

    def test_signal_for_unknown_task_is_ignored(self):
        self.signal(DONE_SIGNAL, task_name="Paint")
        self.assertIsNone(self.manager.active_task)
        self.output.show_message.assert_not_called()


class ProgressDoneDetectorTests(unittest.TestCase):
    """Connect Cables: done once its progress climbs to 0.60 and falls back to 0.15."""

    def setUp(self):
        self.tracker, _ = build_task_tracking(ROOT, logger=Mock())
        self.detector = ProgressDoneDetector("Connect Cables", high=0.60, low=0.15)

    def feed(self, samples, chunk=4):
        seen = []
        for i in range(0, len(samples), chunk):
            part = samples[i:i + chunk]
            seen += self.detector.update(DetectorContext(part[-1][0], None, None, self.tracker,
                                                         recognition=part))
        return seen

    @staticmethod
    def connecting(values, task="Connect Cables"):
        return [(T0 + 0.2 * i, task, value) for i, value in enumerate(values)]

    def test_one_climb_and_fall_is_done_once(self):
        seen = self.feed(self.connecting([0.0, 0.3, 0.5, 0.65, 0.4, 0.1, 0.3, 0.7, 0.05]))
        self.assertEqual([(s.task_name, s.piece_id, s.name) for s in seen],
                         [("Connect Cables", 1, DONE_SIGNAL)])
        self.assertEqual(seen[0].details["why"], "progress fell back")
        self.assertEqual(self.detector.name, "connect cables done")

    def test_jitter_below_the_climb_is_nothing(self):
        self.assertEqual(self.feed(self.connecting([0.1, 0.4, 0.55, 0.2, 0.05, 0.5, 0.1])), [])

    def test_recognition_moving_on_after_the_climb_is_done_too(self):
        seen = self.feed(self.connecting([0.2, 0.62]) + [(T0 + 1.0, "Clamp Coupling", 0.1)])
        self.assertEqual(seen[0].details["why"], "recognition moved on to Clamp Coupling")

    def test_nothing_once_the_task_is_done(self):
        self.tracker.confirm_done("Connect Cables", piece_id=1)
        self.assertEqual(self.feed(self.connecting([0.0, 0.7, 0.1])), [])


class DurationDoneDetectorTests(unittest.TestCase):
    """Clamp Coupling: done by time after it (or Connect Cables before it) got going, or
    once the model shows what follows it -- recognition hardly ever shows it itself."""

    def setUp(self):
        self.tracker, _ = build_task_tracking(ROOT, logger=Mock())
        self.detector = DurationDoneDetector("Clamp Coupling", limit_s=14.0, after_task="Connect Cables",
                                             next_tasks=("Pull Cables", "Lift"), confirm_events=3)

    def update(self, now, model_steps=()):
        return self.detector.update(DetectorContext(now, None, None, self.tracker,
                                                    model_steps=list(model_steps)))

    def connected(self) -> float:
        self.tracker.confirm_done("Connect Cables", piece_id=1)
        return self.tracker.get("Connect Cables", 1).finished_at

    def test_done_by_time_after_connect_cables_without_clamp_coupling_ever_recognized(self):
        done_at = self.connected()
        self.assertEqual(self.update(done_at + 13.9), [])
        seen = self.update(done_at + 14.1)
        self.assertEqual([(s.task_name, s.piece_id, s.name) for s in seen],
                         [("Clamp Coupling", 1, DONE_SIGNAL)])
        self.assertEqual(seen[0].details["why"], "14.0 s since it started")
        self.assertEqual(self.update(done_at + 30.0), [])  # once

    def test_done_once_the_model_shows_the_next_panels_task_three_times_running(self):
        done_at = self.connected()
        self.assertEqual(self.update(done_at + 1, [(0, "Pull Cables"), (1, "Pull Cables"),
                                                   (2, "Screw"), (3, "Pull Cables")]), [])
        seen = self.update(done_at + 2, [(4, "Lift"), (5, "Pull Cables")])
        self.assertEqual(seen[0].details["why"], "recognition shows Pull Cables, which follows it")

    def test_nothing_before_it_or_connect_cables_got_going(self):
        self.assertEqual(self.update(T0 + 1e6, [(t, "Pull Cables") for t in range(5)]), [])

    def test_its_own_start_counts_too(self):
        self.tracker.start_task("Clamp Coupling", 1)  # closes Connect Cables too, at the same time
        started = self.tracker.get("Clamp Coupling", 1).started_at
        self.assertEqual(self.update(started + 13.9), [])
        self.assertEqual(self.update(started + 14.1)[0].name, DONE_SIGNAL)

    def test_nothing_once_the_task_is_done(self):
        done_at = self.connected()
        self.tracker.confirm_done("Clamp Coupling", piece_id=1)
        self.assertEqual(self.update(done_at + 100.0, [(t, "Lift") for t in range(5)]), [])



class ClampCouplingDoneTests(unittest.TestCase):
    """Clamp Coupling: done by its own progress rising and falling back, as recognition
    showed it in run 2026-10-01 09:59 -- not by time, nor by the next panel's task."""

    # Clamp Coupling's progress over one recognized clamp (run 2026-10-01 09:59, 10:03:59-10:04:14).
    CLAMPING = [0.50, 0.53, 0.55, 0.57, 0.54, 0.51, 0.47, 0.44, 0.41, 0.37, 0.30, 0.22, 0.18]

    def setUp(self):
        self.tracker, _ = build_task_tracking(ROOT, logger=Mock())
        self.detector = next(d for d in build_detectors().detectors if d.name == "clamp coupling done")
        self.t = T0

    def feed(self, steps):
        """steps: (task, progress) in order, one update each, 0.2 s apart."""
        samples = []
        for task, progress in steps:
            self.t += 0.2
            samples.append((self.t, task, progress))
        return self.detector.update(DetectorContext(self.t, None, None, self.tracker, recognition=samples))

    def clamping(self, values=None):
        return [("Clamp Coupling", value) for value in (values or self.CLAMPING)]

    def test_built_on_its_progress(self):
        self.assertIsInstance(self.detector, ProgressDoneDetector)
        self.assertEqual((self.detector.high, self.detector.low, self.detector.moved_on_updates),
                         (config.CLAMP_COUPLING_DONE_HIGH, config.CLAMP_COUPLING_DONE_LOW,
                          config.SEQUENCE_CONFIRM_EVENTS))

    def test_a_recorded_clamp_is_done_once_its_progress_falls_back(self):
        self.tracker.confirm_done("Connect Cables", piece_id=1)
        seen = self.feed(self.clamping())
        self.assertEqual([(s.task_name, s.piece_id, s.name) for s in seen],
                         [("Clamp Coupling", 1, DONE_SIGNAL)])
        self.assertEqual(seen[0].details["why"], "progress fell back")

    def test_time_idle_and_the_next_panels_pull_cables_never_end_it(self):
        """Run 2026-10-01 10:30: done 14.2 s after Connect Cables with nobody clamping,
        and again whenever the operator reset it while the human was on Pull Cables."""
        self.tracker.confirm_done("Connect Cables", piece_id=1)
        self.assertEqual(self.feed([("Non Related Task", 0.0)] * 300), [])  # a minute idle
        self.assertEqual(self.feed([("Pull Cables", 0.3)] * 50), [])
        self.assertNotEqual(self.tracker.get("Clamp Coupling", 1).status, T.DONE)

    def test_a_flicker_does_not_end_it_but_moving_on_does(self):
        self.tracker.confirm_done("Connect Cables", piece_id=1)
        steps = self.clamping([0.50, 0.53]) + [("Pull Cables", 0.3)] * 2 + self.clamping([0.52])
        self.assertEqual(self.feed(steps), [])  # two updates elsewhere: still clamping
        seen = self.feed([("Pull Cables", 0.3)] * 3)
        self.assertEqual(seen[0].details["why"], "recognition moved on to Pull Cables")

    def test_set_back_to_not_done_it_waits_for_a_new_clamp(self):
        self.tracker.confirm_done("Connect Cables", piece_id=1)
        self.assertEqual(len(self.feed(self.clamping())), 1)
        self.tracker.confirm_done("Clamp Coupling", piece_id=1)  # as TaskManager does with the signal
        self.assertEqual(self.feed([("Pull Cables", 0.3)] * 5), [])  # the human is on piece 2 now
        self.tracker.set_manually("Clamp Coupling", 1, T.NOT_DONE)  # the operator: not clamped yet
        self.assertEqual(self.tracker.human_piece_id, 1)
        self.assertEqual(self.feed([("Pull Cables", 0.3)] * 20), [])  # not done again at once
        self.assertEqual(len(self.feed(self.clamping())), 1)


class RuntimeWiringTests(unittest.TestCase):
    def make_system(self):
        from communication_runtime import HRCSystem

        system = HRCSystem.__new__(HRCSystem)
        system.event_queue = EventQueue()
        system.logger = Mock()
        system.ros = Mock(drain_wrench=Mock(return_value=[(T0, [0.0] * 6)]))
        system.task_manager = SimpleNamespace(active_task=None, held_piece_id=None, tracker=None)
        system.detectors = Mock(update=Mock(return_value=[Signal("Screw", 1, SCREW_COUNT, 0.5, "force screw")]))
        return system

    def test_log_mode_only_logs_and_on_mode_queues_task_signals(self):
        system = self.make_system()
        with patch.object(config, "TASK_DETECTORS_MODE", "log"):
            system._run_detectors()
        self.assertTrue(system.event_queue.empty())
        self.assertIn("log only", system.logger.log_message.call_args.args[0])
        self.assertEqual(system.detectors.update.call_args.args[0].wrench, [(T0, [0.0] * 6)])
        with patch.object(config, "TASK_DETECTORS_MODE", "on"):
            system._run_detectors()
        event = system.event_queue.get()
        self.assertEqual((event.event_type, event.source), (E.TASK_SIGNAL, "detector:force screw"))
        self.assertEqual(event.payload, {"task_name": "Screw", "piece_id": 1, "signal": SCREW_COUNT, "value": 0.5})

    def test_recognition_updates_reach_the_detectors_next_pass(self):
        system = self.make_system()
        # No step choice (an update without scores, or no tracker): the model's own step.
        system.task_manager = Mock(active_task=None, held_piece_id=None, tracker=None,
                                   last_recognition=None)
        system.communication = Mock()
        system.detectors.update.return_value = []
        system.event_queue.put(Event(E.HUMAN_TASK_UPDATE, "recognition", timestamp=5.0, payload={
            "step_id": config.STEP_NAMES.index("Screw"), "progress": 0.62, "round_id": 0}))
        system.event_queue.put(Event(E.HUMAN_LOCATION_UPDATE, "recognition", payload={"x": 0}))
        system.process_events()
        self.assertEqual(system.detectors.update.call_args.args[0].recognition, [])
        system.task_manager.handle_event.assert_called()  # TaskManager still gets it too
        system.process_events()
        context = system.detectors.update.call_args.args[0]
        self.assertEqual(context.recognition, [(5.0, "Screw", 0.62)])
        self.assertEqual(context.model_steps, [(5.0, "Screw")])
        system.process_events()
        self.assertEqual(system.detectors.update.call_args.args[0].recognition, [])

    def test_detectors_follow_the_step_the_sequence_chose(self):
        system = self.make_system()
        system.communication = Mock()
        system.detectors.update.return_value = []
        choices = iter([StepChoice("Screw", 0.55, "Pull Cables", "by the sequence"),
                        StepChoice(None, 0.0, "Place", "no plausible task scores high enough")])
        system.task_manager = Mock(active_task=None, held_piece_id=None, tracker=None)
        system.task_manager.handle_event.side_effect = (
            lambda event: setattr(system.task_manager, "last_recognition", next(choices)))
        for t, step in ((5.0, "Pull Cables"), (6.0, "Place")):
            system.event_queue.put(Event(E.HUMAN_TASK_UPDATE, "recognition", timestamp=t, payload={
                "step_id": config.STEP_NAMES.index(step), "progress": 0.1, "round_id": 0}))
        system.process_events()
        system.process_events()
        context = system.detectors.update.call_args.args[0]
        self.assertEqual(context.recognition, [(5.0, "Screw", 0.55), (6.0, "Non Related Task", 0.0)])
        self.assertEqual(context.model_steps, [(5.0, "Pull Cables"), (6.0, "Place")])

    def test_screw_progress_cycles_end_the_holding_through_the_runtime(self):
        from communication_runtime import HRCSystem
        from task_transition_detector import build_detectors

        system = HRCSystem.__new__(HRCSystem)
        system.event_queue = EventQueue()
        system.logger = Mock()
        system.ros = Mock(drain_wrench=Mock(return_value=[]))
        system.communication = Mock()
        output, parser = Mock(), CommandParser()
        tracker, policy = build_task_tracking(ROOT, logger=Mock())
        system.task_manager = TaskManager(StateMachine(), PendingTaskPool(), Mock(), MessageManager(), output,
                                          GHDispatcher(Mock()), Mock(), Mock(),
                                          task_tracker=tracker, trigger_policy=policy)
        system.detectors = build_detectors()

        def step(*events):
            for event in events:
                system.event_queue.put(event)
            system.process_events()

        screw = config.STEP_NAMES.index("Screw")
        with patch.object(config, "TASK_DETECTORS_MODE", "on"), patch.object(config, "SCREWS_PER_PANEL", 6):
            step(Event(E.HUMAN_TASK_UPDATE, "recognition", payload={
                "step_id": config.STEP_NAMES.index("Pull Cables"), "progress": 0.8, "round_id": 0}))
            step(parser.parse("yes"), Event(E.ROBOT_RUNNING, "ros"), Event(E.ROBOT_SUCCESS, "ros"),
                 parser.parse("adjustment done"))  # free drive came on at arrival
            self.assertEqual(system.task_manager.active_task.state, S.R_HOLDING)
            progress_seen = []
            for t, name, progress in screwing([CLIMB] * 6):
                step(Event(E.HUMAN_TASK_UPDATE, "recognition", timestamp=t, payload={
                    "step_id": config.STEP_NAMES.index(name), "progress": progress, "round_id": 0}))
                progress_seen.append(tracker.get("Screw", 1).progress)
                if round(progress_seen[-1] * 6) == 3:
                    self.assertIn("Bring Connector", [e["task_name"] for e in system.task_manager.waiting_triggers])
            step()  # the last update's signals
        self.assertEqual(sorted({round(p * 6) for p in progress_seen}), [0, 1, 2, 3, 4, 5])
        leave = system.task_manager.active_task
        self.assertEqual((leave.task_id, leave.state), (config.TASK_LEAVE, S.R_WAITING_RESPONSE))
        self.assertEqual(tracker.get("Screw", 1).status, T.WORKING)  # recognition or "screw done" decides
        self.assertIn("The panel looks secured.", [c.args[0] for c in output.show_message.call_args_list])
        self.assertNotEqual(tracker.get("Connect Cables", 1).status, T.WORKING)  # ignored while holding

    def test_clamp_coupling_ends_by_its_progress_not_by_the_next_panels_task(self):
        """While piece 1's Clamp Coupling is open, the model showing the next panel's Pull
        Cables does not end it (run 2026-10-01 10:30); its own progress rising and falling
        back does -- and the human then moves on to piece 2's Pull Cables."""
        from communication_runtime import HRCSystem

        system = HRCSystem.__new__(HRCSystem)
        system.event_queue = EventQueue()
        system.logger = Mock()
        system.ros = Mock(drain_wrench=Mock(return_value=[]))
        system.communication = Mock()
        tracker, policy = build_task_tracking(ROOT, logger=Mock())
        system.task_manager = TaskManager(StateMachine(), PendingTaskPool(), Mock(), MessageManager(), Mock(),
                                          GHDispatcher(Mock()), Mock(), Mock(),
                                          task_tracker=tracker, trigger_policy=policy)
        system.detectors = build_detectors()
        tracker.confirm_done("Connect Cables", piece_id=1)

        def update(step, progress):
            index = config.STEP_NAMES.index(step)
            scores = [0.02] * len(config.STEP_NAMES)
            scores[index] = 0.8
            lanes = [0.0] * len(config.STEP_NAMES)
            lanes[index] = progress
            system.event_queue.put(Event(E.HUMAN_TASK_UPDATE, "recognition", payload={
                "step_id": index, "progress": progress, "round_id": 0,
                "step_probabilities": scores, "step_progress": lanes}))
            system.process_events()

        with patch.object(config, "TASK_DETECTORS_MODE", "on"):
            for _ in range(10):
                update("Pull Cables", 0.3)
                self.assertIsNone(system.task_manager.last_recognition.task_name)  # filtered
            system.process_events()
            self.assertNotEqual(tracker.get("Clamp Coupling", 1).status, T.DONE)
            self.assertEqual(tracker.human_piece_id, 1)
            for progress in ClampCouplingDoneTests.CLAMPING:
                update("Clamp Coupling", progress)
            system.process_events()  # the detectors' pass sees the fall
            system.process_events()  # their done signal is handled
            self.assertEqual(tracker.get("Clamp Coupling", 1).status, T.DONE)
            self.assertEqual(tracker.human_piece_id, 2)
            for _ in range(3):  # confirm_events: a switch away from the reference task
                update("Pull Cables", 0.3)
        self.assertEqual((tracker.reference_task, tracker.reference_piece_id), ("Pull Cables", 2))

    def test_ros_buffers_wrench_until_drained(self):
        ros = ROSCommunication(auto_connect=False, wrench_topic="/UR10e/TCPForce/live")
        ros._on_wrench_message({"wrench": {"force": {"x": 1, "y": 2, "z": 3},
                                           "torque": {"x": 4, "y": 5, "z": 6}}})
        ros._on_wrench_message({"wrench": {"force": {"x": 1}}})  # malformed: dropped
        samples = ros.drain_wrench()
        self.assertEqual([wrench for _t, wrench in samples], [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        self.assertEqual(ros.drain_wrench(), [])


if __name__ == "__main__":
    unittest.main()

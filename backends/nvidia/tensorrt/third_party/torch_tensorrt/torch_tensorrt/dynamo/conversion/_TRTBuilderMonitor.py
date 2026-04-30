import os
import sys
from typing import Any, Dict, Optional

import tensorrt as trt


class _ASCIIMonitor(trt.IProgressMonitor):  # type: ignore
    def __init__(self, engine_name: str = "") -> None:
        trt.IProgressMonitor.__init__(self)
        self._active_phases: Dict[str, Dict[str, Any]] = {}
        self._step_result = True

        self._render = True
        if (ci_env_var := os.environ.get("CI_BUILD")) is not None:
            if ci_env_var == "1":
                self._render = False

    def phase_start(
        self, phase_name: str, parent_phase: Optional[str], num_steps: int
    ) -> None:
        try:
            if parent_phase is not None:
                nbIndents = 1 + self._active_phases[parent_phase]["nbIndents"]
            else:
                nbIndents = 0
            self._active_phases[phase_name] = {
                "title": phase_name,
                "steps": 0,
                "num_steps": num_steps,
                "nbIndents": nbIndents,
            }
            self._redraw()
        except KeyboardInterrupt:
            _step_result = False

    def phase_finish(self, phase_name: str) -> None:
        try:
            del self._active_phases[phase_name]
            self._redraw(blank_lines=1)  # Clear the removed phase.
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name: str, step: int) -> bool:
        try:
            self._active_phases[phase_name]["steps"] = step
            self._redraw()
            return self._step_result
        except KeyboardInterrupt:
            return False

    def _redraw(self, *, blank_lines: int = 0) -> None:
        if self._render:

            def clear_line() -> None:
                print("\x1b[2K", end="")

            def move_to_start_of_line() -> None:
                print("\x1b[0G", end="")

            def move_cursor_up(lines: int) -> None:
                print("\x1b[{}A".format(lines), end="")

            def progress_bar(steps: int, num_steps: int) -> str:
                INNER_WIDTH = 10
                completed_bar_chars = int(INNER_WIDTH * steps / float(num_steps))
                return "[{}{}]".format(
                    "=" * completed_bar_chars, "-" * (INNER_WIDTH - completed_bar_chars)
                )

            # Set max_cols to a default of 200 if not run in interactive mode.
            max_cols = os.get_terminal_size().columns if sys.stdout.isatty() else 200

            move_to_start_of_line()
            for phase in self._active_phases.values():
                phase_prefix = "{indent}{bar} {title}".format(
                    indent=" " * phase["nbIndents"],
                    bar=progress_bar(phase["steps"], phase["num_steps"]),
                    title=phase["title"],
                )
                phase_suffix = "{steps}/{num_steps}".format(**phase)
                allowable_prefix_chars = max_cols - len(phase_suffix) - 2
                if allowable_prefix_chars < len(phase_prefix):
                    phase_prefix = phase_prefix[0 : allowable_prefix_chars - 3] + "..."
                clear_line()
                print(phase_prefix, phase_suffix)
            for line in range(blank_lines):
                clear_line()
                print()
            move_cursor_up(len(self._active_phases) + blank_lines)
            sys.stdout.flush()


try:
    from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn

    class _RichMonitor(trt.IProgressMonitor):  # type: ignore
        def __init__(self, engine_name: str = "") -> None:
            trt.IProgressMonitor.__init__(self)
            self._active_phases: Dict[str, TaskID] = {}
            self._step_result = True

            self._progress_monitors = Progress(
                TextColumn("  "),
                TimeElapsedColumn(),
                TextColumn("{task.description}: "),
                BarColumn(),
                TextColumn(" {task.percentage:.0f}% ({task.completed}/{task.total})"),
            )

            self._render = True
            if (ci_env_var := os.environ.get("CI_BUILD")) is not None:
                if ci_env_var == "1":
                    self._render = False

            if self._render:
                self._progress_monitors.start()

        def phase_start(
            self, phase_name: str, parent_phase: Optional[str], num_steps: int
        ) -> None:
            try:
                self._active_phases[phase_name] = self._progress_monitors.add_task(
                    phase_name, total=num_steps
                )
                self._progress_monitors.refresh()
            except KeyboardInterrupt:
                # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
                _step_result = False

        def phase_finish(self, phase_name: str) -> None:
            try:
                self._progress_monitors.update(
                    self._active_phases[phase_name], visible=False
                )
                self._progress_monitors.stop_task(self._active_phases[phase_name])
                self._progress_monitors.remove_task(self._active_phases[phase_name])
                self._progress_monitors.refresh()
            except KeyboardInterrupt:
                _step_result = False

        def step_complete(self, phase_name: str, step: int) -> bool:
            try:
                self._progress_monitors.update(
                    self._active_phases[phase_name], completed=step
                )
                self._progress_monitors.refresh()
                return self._step_result
            except KeyboardInterrupt:
                # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
                return False

        def __del__(self) -> None:
            if self._progress_monitors:
                self._progress_monitors.stop()

    TRTBulderMonitor: trt.IProgressMonitor = _RichMonitor
except ImportError:
    TRTBulderMonitor: trt.IProgressMonitor = _ASCIIMonitor  # type: ignore[no-redef]

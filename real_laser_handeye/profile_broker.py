from __future__ import annotations

import threading
import time

from .hardware import LaserInterface, ProfileSample


class ProfileBroker(LaserInterface):
    """Single-owner acquisition broker shared by live view and capture.

    Only the broker thread calls the vendor adapter. ``latest`` is non-blocking
    and intended for UI preview. ``capture_profile`` waits for a frame whose
    sequence is newer than the call, so a stationary capture never reuses the
    profile shown before the robot settled.
    """

    def __init__(
        self,
        laser: LaserInterface,
        *,
        acquisition_timeout_s: float = 1.0,
        retry_delay_s: float = 0.02,
        capture_discard_frames: int = 1,
        maximum_rate_hz: float = 50.0,
    ) -> None:
        self.laser = laser
        self.acquisition_timeout_s = float(acquisition_timeout_s)
        self.retry_delay_s = float(retry_delay_s)
        self.capture_discard_frames = int(capture_discard_frames)
        if self.capture_discard_frames < 0:
            raise ValueError("capture_discard_frames must be non-negative")
        if maximum_rate_hz <= 0.0:
            raise ValueError("maximum_rate_hz must be positive")
        self.minimum_period_s = 1.0 / float(maximum_rate_hz)
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self._latest: ProfileSample | None = None
        self._last_error: str | None = None

    def connect(self) -> None:
        self.start()

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._acquisition_loop,
                name="laser-profile-broker",
                daemon=True,
            )
            self._thread.start()

    def close(self) -> None:
        self.stop()

    def stop(self, timeout_s: float | None = None) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            wait_s = (
                self.acquisition_timeout_s + 0.5
                if timeout_s is None
                else float(timeout_s)
            )
            thread.join(timeout=wait_s)
        if thread is not None and thread.is_alive():
            raise TimeoutError(
                "profile acquisition thread did not stop; vendor laser must not be closed yet"
            )
        self._thread = None

    def _acquisition_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                sample = self.laser.capture_profile(
                    timeout_s=self.acquisition_timeout_s
                )
                # Publish an immutable frame. The vendor's callback buffer is
                # already copied by the adapter and ProfileSample copies again.
                published = ProfileSample(sample.points_s, sample.timestamp_ns)
                published.points_s.setflags(write=False)
                with self._condition:
                    self._sequence += 1
                    self._latest = published
                    self._last_error = None
                    self._condition.notify_all()
                remaining = self.minimum_period_s - (time.monotonic() - started)
                if remaining > 0.0:
                    self._stop_event.wait(remaining)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                with self._condition:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._condition.notify_all()
                self._stop_event.wait(self.retry_delay_s)

    def latest(self) -> tuple[int, ProfileSample | None, str | None]:
        with self._condition:
            return self._sequence, self._latest, self._last_error

    def capture_profile(self, *, timeout_s: float) -> ProfileSample:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            start_sequence = self._sequence
            required_sequence = start_sequence + self.capture_discard_frames + 1
            while self._sequence < required_sequence:
                if self._stop_event.is_set():
                    raise RuntimeError("profile broker is stopped")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    detail = "" if self._last_error is None else f"; last error: {self._last_error}"
                    raise TimeoutError(f"timed out waiting for a fresh laser profile{detail}")
                self._condition.wait(timeout=remaining)
            assert self._latest is not None
            return self._latest

"""
GenerationTimer - Tracks timing for different phases of image generation

This module provides timing utilities for the Stable Diffusion WebUI Forge Neo
generation pipeline. It tracks both batch-level and per-image timing data.
"""

from __future__ import annotations
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set
from contextlib import contextmanager


class PhaseTimer:
    """Internal timer for tracking phases"""

    def __init__(self, name: str = ""):
        self.name = name
        self.timings = defaultdict(float)
        self.current_phase = None
        self.current_phase_start = None
        self.total_start = None
        self.phases_seen = set()

    def start(self, phase: str, log: bool = True):
        """Start timing a phase. Automatically stops the previous phase if one is running."""
        now = time.perf_counter()

        # Initialize total timer on first start
        if self.total_start is None:
            self.total_start = now

        # Stop the previous phase if one is running
        if self.current_phase is not None and self.current_phase_start is not None:
            elapsed = now - self.current_phase_start
            self.timings[self.current_phase] += elapsed
            self.phases_seen.add(self.current_phase)
            if log:
                prefix = f"[{self.name}] " if self.name else ""
                print(f"[Timing] {prefix}{self.current_phase}: {elapsed:.2f}s")

        # Start the new phase
        self.current_phase = phase
        self.current_phase_start = now
        if log:
            prefix = f"[{self.name}] " if self.name else ""
            print(f"[Timing] {prefix}Starting: {phase}")

    def stop(self, log: bool = True) -> float:
        """Stop timing the current phase and return elapsed time."""
        if self.current_phase is None or self.current_phase_start is None:
            return 0

        now = time.perf_counter()
        elapsed = now - self.current_phase_start
        self.timings[self.current_phase] += elapsed
        self.phases_seen.add(self.current_phase)

        if log:
            prefix = f"[{self.name}] " if self.name else ""
            print(f"[Timing] {prefix}{self.current_phase}: {elapsed:.2f}s")

        self.current_phase = None
        self.current_phase_start = None

        return elapsed

    def add_time(self, phase: str, duration: float):
        """Add time to a phase from an external source."""
        self.timings[phase] += duration
        self.phases_seen.add(phase)

    def get_total_time(self) -> float:
        """Get total time since first start() call"""
        if self.total_start is None:
            return 0
        return time.perf_counter() - self.total_start

    def get_sum_of_phases(self) -> float:
        """Get sum of all phase timings"""
        return sum(self.timings.values())


class GenerationTimer:
    """Tracks timing for different phases of image generation"""

    def __init__(self):
        # Batch-level timer (persists across all images)
        self.batch_timer = PhaseTimer("Batch")

        # Array of individual image timers
        self.image_timers: List[PhaseTimer] = []

        # Current image timer (active)
        self.current_image_timer = PhaseTimer("Image 0")
        self.image_timers.append(self.current_image_timer)

        # Per-image tracking (legacy support)
        self.image_timings = []
        self.batch_start_time = None

    def start(self, phase: str):
        """
        Start timing a phase on both batch and current image timer.
        Automatically stops the previous phase if one is running.

        Args:
            phase: Name of the phase to start timing
        """
        # Start on both timers
        # Only log for image timer to avoid duplicate logs
        self.batch_timer.start(phase, log=False)
        self.current_image_timer.start(phase, log=True)

    def stop(self) -> float:
        """
        Stop timing the current phase on both timers and return elapsed time from image timer.

        Returns:
            Elapsed time for the stopped phase, or 0 if no phase was running
        """
        self.batch_timer.stop(log=False)
        return self.current_image_timer.stop(log=True)

    @contextmanager
    def time_phase(self, phase: str):
        """
        Context manager for timing a phase automatically.

        Usage:
            with timer.time_phase('sampling'):
                # do sampling work
                pass
        """
        self.start(phase)
        try:
            yield
        finally:
            self.stop()

    def add_time(self, phase: str, duration: float):
        """Add time to a phase from an external source (e.g., memory management)"""
        self.batch_timer.add_time(phase, duration)
        self.current_image_timer.add_time(phase, duration)

    def get_total_time(self) -> float:
        """Get total time for current image since first start() call"""
        return self.current_image_timer.get_total_time()

    def get_batch_total_time(self) -> float:
        """Get total time for the batch"""
        return self.batch_timer.get_total_time()

    def reset_image_timer(self):
        """Create a new image timer for the next image (batch timer continues)"""
        # Stop current phase on current timer
        self.current_image_timer.stop(log=True)

        # Create new timer
        image_index = len(self.image_timers)
        self.current_image_timer = PhaseTimer(f"Image {image_index}")
        self.image_timers.append(self.current_image_timer)

    def reset_batch_timer(self):
        """Reset everything for a new batch"""
        self.batch_timer = PhaseTimer("Batch")
        self.image_timers = []
        self.current_image_timer = PhaseTimer("Image 0")
        self.image_timers.append(self.current_image_timer)

    def get_image_count(self) -> int:
        """Get number of images processed"""
        return len(self.image_timers)

    @property
    def timings(self) -> Dict[str, float]:
        """
        Property for backward compatibility.
        Returns the batch timer's timings dict.
        """
        return self.batch_timer.timings

    def get_batch_average(self, phase: str) -> float:
        """Get average time for a phase across all images in the batch"""
        count = self.get_image_count()
        if count == 0:
            return 0
        return self.batch_timer.timings[phase] / count

    def get_phases_in_order(self, timings_dict: Dict[str, float]) -> List[str]:
        """
        Get phases in a sensible display order.
        Known phases first in logical order, then alphabetically sorted unknowns.
        """
        # Common phase order
        known_phases = [
            'initialization',
            'model_loading',
            'scripts_process',
            'memory_management',
            'text_encoding',
            'kmodel_load',
            'sampling',
            'kmodel_load_base',
            'sampling_base',
            'vae_decoding',
            'vae_decoding_base',
            'kmodel_load_hires',
            'sampling_hires',
            'vae_decoding_hires',
            'post_processing',
            'save_images',
        ]

        # Filter to only phases that exist in timings
        result = [p for p in known_phases if p in timings_dict and timings_dict[p] > 0]

        # Add any unknown phases alphabetically
        unknown = sorted(set(timings_dict.keys()) - set(known_phases))
        result.extend(p for p in unknown if timings_dict[p] > 0)

        return result

    def categorize_phase(self, phase: str) -> str:
        """Determine which category a phase belongs to"""
        if 'hires' in phase:
            return 'hires'
        elif phase in ['post_processing', 'save_images']:
            return 'post'
        else:
            return 'main'

    def format_summary(self, title: str = "GENERATION COMPLETE", image_index: Optional[int] = None) -> str:
        """
        Format timing summary in the new style.

        Args:
            title: Title for the summary header
            image_index: If provided, show summary for that image; otherwise use current image

        Returns:
            Formatted timing summary string
        """
        # Stop any running phase to get accurate totals
        self.stop()

        # Get the timer to display
        if image_index is not None and 0 <= image_index < len(self.image_timers):
            timer = self.image_timers[image_index]
        else:
            timer = self.current_image_timer

        timings_dict = timer.timings
        total = sum(timings_dict.values())

        if total == 0:
            return ""

        lines = []
        lines.append("=" * 60)
        lines.append(f"[Timing] {title} - Total time: {total:.2f}s")
        lines.append("=" * 60)

        # Get phases in order
        phases = self.get_phases_in_order(timings_dict)

        # Group phases by category
        main_phases = []
        hires_phases = []
        post_phases = []

        for phase in phases:
            category = self.categorize_phase(phase)
            if category == 'hires':
                hires_phases.append(phase)
            elif category == 'post':
                post_phases.append(phase)
            else:
                main_phases.append(phase)

        # Display main phases
        if main_phases:
            lines.append("[Timing] Phase breakdown:")
            for phase in main_phases:
                duration = timings_dict[phase]
                pct = (duration / total * 100) if total > 0 else 0
                lines.append(f"  {phase:20s}: {duration:6.2f}s ({pct:5.1f}%)")

        # Display hires fix phases
        if hires_phases:
            lines.append("")
            lines.append("[Timing] Hires fix phases:")
            for phase in hires_phases:
                duration = timings_dict[phase]
                pct = (duration / total * 100) if total > 0 else 0
                # Format display name
                display_name = phase.replace('_hires', ' (hires)')
                lines.append(f"  {display_name:20s}: {duration:6.2f}s ({pct:5.1f}%)")

        # Display post-processing phases
        if post_phases:
            lines.append("")
            for phase in post_phases:
                duration = timings_dict[phase]
                pct = (duration / total * 100) if total > 0 else 0
                lines.append(f"  {phase:20s}: {duration:6.2f}s ({pct:5.1f}%)")

        return "\n".join(lines)

    def format_batch_summary(self) -> str:
        """Format batch-level timing summary with per-image breakdown"""
        image_count = self.get_image_count()
        if image_count == 0:
            return ""

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"[Timing] BATCH SUMMARY - {image_count} image(s)")
        lines.append("=" * 60)

        # Show totals and averages
        total = self.batch_timer.get_sum_of_phases()
        lines.append(f"[Timing] Total batch time: {total:.2f}s")
        lines.append(f"[Timing] Average per image: {total / image_count:.2f}s")
        lines.append("")

        # Get phases in order
        phases = self.get_phases_in_order(self.batch_timer.timings)

        # Group phases by category
        main_phases = []
        hires_phases = []
        post_phases = []

        for phase in phases:
            category = self.categorize_phase(phase)
            if category == 'hires':
                hires_phases.append(phase)
            elif category == 'post':
                post_phases.append(phase)
            else:
                main_phases.append(phase)

        # Display main phases
        if main_phases:
            lines.append("[Timing] Phase totals:")
            for phase in main_phases:
                duration = self.batch_timer.timings[phase]
                avg = self.get_batch_average(phase)
                pct = (duration / total * 100) if total > 0 else 0
                lines.append(f"  {phase:20s}: {duration:6.2f}s (avg: {avg:5.2f}s, {pct:5.1f}%)")

        # Display hires fix phases
        if hires_phases:
            lines.append("")
            lines.append("[Timing] Hires fix totals:")
            for phase in hires_phases:
                duration = self.batch_timer.timings[phase]
                avg = self.get_batch_average(phase)
                pct = (duration / total * 100) if total > 0 else 0
                display_name = phase.replace('_hires', ' (hires)')
                lines.append(f"  {display_name:20s}: {duration:6.2f}s (avg: {avg:5.2f}s, {pct:5.1f}%)")

        # Display post-processing phases
        if post_phases:
            lines.append("")
            for phase in post_phases:
                duration = self.batch_timer.timings[phase]
                avg = self.get_batch_average(phase)
                pct = (duration / total * 100) if total > 0 else 0
                lines.append(f"  {phase:20s}: {duration:6.2f}s (avg: {avg:5.2f}s, {pct:5.1f}%)")

        # Show per-image breakdown
        if image_count > 1:
            lines.append("")
            lines.append("[Timing] Per-image breakdown:")
            for i, img_timer in enumerate(self.image_timers):
                img_total = img_timer.get_sum_of_phases()
                lines.append(f"  Image {i}: {img_total:.2f}s")

        return "\n".join(lines)

    # ========================================================================
    # Legacy methods for backward compatibility
    # ========================================================================

    def reset_current_generation(self):
        """Reset timing for the current generation (legacy - now resets image timer)"""
        self.reset_image_timer()

    def accumulate_to_batch(self):
        """Accumulate to batch (legacy - now a no-op since batch timer runs continuously)"""
        # This is now a no-op since batch timer runs continuously
        pass

    def add_image_timing(self, index: int, phase_timings: Dict[str, float]):
        """Add timing info for a specific image (legacy support)"""
        timing_dict = {'index': index}
        timing_dict.update(phase_timings)
        self.image_timings.append(timing_dict)

    def get_image_total_time(self, index: int) -> float:
        """Get total time for a specific image"""
        if 0 <= index < len(self.image_timers):
            return self.image_timers[index].get_sum_of_phases()
        return 0

    def get_batch_total_time_legacy(self) -> float:
        """Get total time for the entire batch using legacy method"""
        total = 0
        for img_timing in self.image_timings:
            total += sum(v for k, v in img_timing.items() if k not in ['index', 'save_image'])
        return total

    def get_aggregate_post_processing_time(self) -> float:
        """Get total time spent on post-processing across all images (legacy support)"""
        total = 0
        for img_timing in self.image_timings:
            total += img_timing.get('total', 0)
        return total

    def get_aggregate_save_time(self) -> float:
        """Get total time spent saving images (legacy support)"""
        total = 0
        for img_timing in self.image_timings:
            total += img_timing.get('save_image', 0)
        return total

    def format_for_metadata(self, use_batch: bool = False) -> dict:
        """
        Format timing data for image metadata.

        Args:
            use_batch: If True, use batch timer; if False, use current image timer

        Returns:
            Dictionary with timing keys appropriate for the context
        """
        timer = self.batch_timer if use_batch else self.current_image_timer
        img_timings = timer.timings
        img_total = timer.get_sum_of_phases()

        result = {}

        if img_total > 0:
            # Use appropriate key names based on context
            if use_batch:
                result["Batch time"] = f"{img_total:.2f}s"
            else:
                result["Image time"] = f"{img_total:.2f}s"

            # Build timing breakdown
            timing_parts = []
            phases = self.get_phases_in_order(img_timings)

            for phase in phases:
                if phase in img_timings and img_timings[phase] > 0:
                    duration = img_timings[phase]
                    pct = (duration / img_total * 100) if img_total > 0 else 0
                    timing_parts.append(f"{phase}={duration:.2f}s ({pct:.1f}%)")

            if timing_parts:
                if use_batch:
                    result["Batch timing breakdown"] = ", ".join(timing_parts)
                else:
                    result["Image timing breakdown"] = ", ".join(timing_parts)

        return result

    def format_image_timing(self, image_index: int) -> str:
        """Format timing for a specific image (legacy support)"""
        if image_index < len(self.image_timings):
            img_timing = self.image_timings[image_index]
            total = sum(v for k, v in img_timing.items() if k != 'index')
            lines = [f"Image {img_timing['index']} generation time: {total:.2f}s"]
            for phase, duration in img_timing.items():
                if phase != 'index':
                    lines.append(f"  {phase}: {duration:.2f}s")
            return "\n".join(lines)
        return ""

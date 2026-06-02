import json
import time
from collections import defaultdict

import matplotlib.pyplot as plt


class Profiler:
    def __init__(self, runProfiler=True):
        # store *lists* of elapsed times, not cumulative sums
        self.times = defaultdict(list)
        self._start_stack = {}
        self.runProfiler = runProfiler

    def start(self, label: str):
        """Mark start time of a section."""
        if self.runProfiler:
            self._start_stack[label] = time.perf_counter()

    def stop(self, label: str):
        """Mark end time of a section and record elapsed."""
        if self.runProfiler:
            if label not in self._start_stack:
                raise ValueError(f"Profiler.stop() called without matching start for '{label}'")
            elapsed = time.perf_counter() - self._start_stack[label]
            self.times[label].append(elapsed)
            del self._start_stack[label]

    def report(self, normalize=False):
        """Return dict of per-label timing stats (non-cumulative)."""
        if self.runProfiler:
            report_dict = {}
            for label, time_list in self.times.items():
                n = len(time_list)
                total_time = sum(time_list)
                avg_time = total_time / n if n > 0 else 0.0
                report_dict[label] = {
                    "count": n,
                    "total_time_s": total_time,
                    "avg_time_s": avg_time,
                    "min_time_s": min(time_list) if time_list else 0.0,
                    "max_time_s": max(time_list) if time_list else 0.0,
                }
            if normalize:
                total_all = sum(r["total_time_s"] for r in report_dict.values())
                for r in report_dict.values():
                    r["fraction"] = r["total_time_s"] / total_all if total_all > 0 else 0.0
            return report_dict

    def to_json(self, filename):
        if self.runProfiler:
            with open(filename, "w") as f:
                json.dump(self.report(), f, indent=2)

    def plot(self, title="Profiler Timing Breakdown", filename: str = None):
        """Plot average time per section."""
        if self.runProfiler:        
            report = self.report(normalize=True)
            if not report:
                print("Profiler has no data to plot.")
                return

            methods = list(report.keys())
            total_times = [report[m]["total_time_s"] for m in methods]
            fractions = [report[m].get("fraction", 0) for m in methods]

            # Sort by total time descending
            sorted_indices = sorted(range(len(total_times)), key=lambda i: total_times[i], reverse=True)
            methods_sorted = [methods[i] for i in sorted_indices]
            total_times_sorted = [total_times[i] for i in sorted_indices]
            fractions_sorted = [fractions[i] for i in sorted_indices]

            plt.figure(figsize=(12, 6))
            bars = plt.bar(methods_sorted, total_times_sorted, color="skyblue")

            for bar, frac, t in zip(bars, fractions_sorted, total_times_sorted):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{frac*100:.1f}%\n({t:.4f}s)",
                        ha="center", va="bottom", fontsize=9)

            plt.ylabel("Total Time [s]")
            plt.xlabel("Section / Label")
            plt.title(title)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=300)
                print(f"Profiler chart saved to '{filename}'")
                plt.close()
            else:
                plt.show()
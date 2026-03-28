"""Live dashboard — updates during ALNS solve."""


def create_live_dashboard(customers, depot, vehicles, update_interval_ms=500, figsize=(20, 12)):
    """Create interactive figure. Returns dashboard handle dict."""
    raise NotImplementedError


def live_callback(dashboard, iteration, sol, eval_result, log):
    """ALNS callback: throttled redraw. Pass to solve(callback=...)."""
    raise NotImplementedError


def start_live_dashboard(customers, depot, vehicles):
    """create + plt.ion() + show. Returns handle."""
    raise NotImplementedError


def stop_live_dashboard(dashboard):
    """plt.ioff() + final render."""
    raise NotImplementedError

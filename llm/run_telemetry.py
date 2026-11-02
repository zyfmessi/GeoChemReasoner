"""A minimal publish/subscribe channel for run telemetry (stdlib only, zero project dependencies, safe to import from any layer).

Aggregates runtime events scattered across layers for the frontend's "run monitor" panel:
llm events (per-call prompt/completion token counts and context window, published by
openai_compatible), tool events (tool dispatch), and rag events (literature retrieval);
subscribers aggregate them and expose the results via /api/langgraph_status. Constraints:
publishers are unaware of subscribers (emit is near-zero cost with no subscribers, so
instrumentation can be unconditional); a single subscriber's exception is swallowed and
never harms the main inference flow; multiple threads emit concurrently, with a lock
guarding the subscriber table.
"""
import threading  # Import threading: guard concurrent access to the subscriber table with a lock

_listeners: list = []  # Subscriber callback table; callback signature fn(event: dict) -> None
_lock = threading.Lock()  # Create the global mutex protecting the subscriber table's additions, removals, and snapshots


def subscribe(fn) -> None:  # Register a telemetry listener callback
    """Register a telemetry listener callback (idempotent: registering the same function repeatedly takes effect only once)."""  # Function docstring: idempotent; re-registering the same callback takes effect only once
    with _lock:  # Operate on the subscriber table under the lock
        if fn not in _listeners:  # Append only if not already registered (ensures idempotence)
            _listeners.append(fn)  # Add the callback to the subscriber table


def unsubscribe(fn) -> None:  # Remove a telemetry listener callback
    """Remove a telemetry listener callback (silently ignores unregistered callbacks)."""  # Function docstring: silent, no error, when not registered
    with _lock:  # Operate on the subscriber table under the lock
        if fn in _listeners:  # Remove only if registered (silent when not)
            _listeners.remove(fn)  # Remove the callback from the subscriber table


def emit(event: dict) -> None:  # Publish one telemetry event to all subscribers
    """Publish one telemetry event to all subscribers; a subscriber's exception affects neither the publisher nor the other subscribers."""  # Function docstring: subscriber exceptions affect neither the publisher nor each other
    with _lock:  # Copy the current subscriber table under the lock
        listeners = list(_listeners)  # Snapshot, safe even if callbacks add/remove subscribers
    for fn in listeners:  # Notify the subscribers one by one over the snapshot
        try:  # Wrap the callback call so exceptions do not escape
            fn(event)  # Invoke the subscriber callback with the event
        except Exception:  # Telemetry must never harm the main inference flow
            pass  # Swallow the subscriber exception, never harming the main flow

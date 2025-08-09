# src/events/event_bus.py

from typing import Dict, List, Callable, Any

class Event:
    """
    A generic container for an event, defined only by its name and a
    flexible data payload.
    """
    def __init__(self, name: str, payload: Dict[str, Any]):
        self.name = name
        self.payload = payload

    def __repr__(self) -> str:
        return f"Event(name='{self.name}', payload={self.payload})"

class EventBus:
    """
    A generic, in-memory event bus that routes events based on string names.
    It is completely decoupled from the structure of the event payloads.
    """
    def __init__(self):
        # The dictionary now maps string names to subscriber callbacks
        self.subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_name: str, callback: Callable):
        """Registers a subscriber for a given event name."""
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(callback)
        print(f"✅ Subscriber {callback.__name__} registered for event: '{event_name}'")

    def publish(self, event: Event):
        """Publishes an event to all subscribers registered for its name."""
        if event.name in self.subscribers:
            for callback in self.subscribers[event.name]:
                try:
                    # Pass the whole event object to the subscriber
                    callback(event)
                except Exception as e:
                    print(f"❌ Error in subscriber {callback.__name__}: {e}")

# Create a single, global instance to be used throughout the application
event_bus = EventBus()
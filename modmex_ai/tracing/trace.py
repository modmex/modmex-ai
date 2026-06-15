from __future__ import annotations

from dataclasses import field

from modmex import BaseModel

from modmex_ai.tracing.step import TraceStep
from modmex_ai.observability import ObservabilityEvent, ObservabilityObserver


class Trace(BaseModel):
    steps: list[TraceStep] = field(default_factory=list)
    observers: list[ObservabilityObserver] = field(default_factory=list)
    observer_errors: list[str] = field(default_factory=list)

    def add(self, type: str, name: str, **data) -> None:
        self.steps.append(TraceStep(type=type, name=name, data=data))
        event = ObservabilityEvent(type=type, name=name, data=data)
        for observer in self.observers:
            try:
                observer.on_event(event)
            except Exception as error:
                self.observer_errors.append(str(error))

    def add_observer(self, observer: ObservabilityObserver) -> None:
        self.observers.append(observer)

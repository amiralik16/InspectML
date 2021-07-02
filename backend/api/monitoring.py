from typing import Callable
from prometheus_fastapi_instrumentator.metrics import Info
from prometheus_client import Counter

def total_animal_prediction() -> Callable[[Info], None]:
    METRIC = Counter(
        "total_animal_prediction", 
        "Number of times a certain animal has been reported.", 
        labelnames=("animals",)
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/predict":
            animal = info.response.headers.get("predicted_class")
            if animal:
                METRIC.labels(animal).inc()

    return instrumentation
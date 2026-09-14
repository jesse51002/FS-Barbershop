from datetime import datetime

from hairstyle_creation.models import HairstyleChangeEvent

ATTEMPTS = 5

def add_to_embedding_queue(event: HairstyleChangeEvent) -> None:
    event.embedding_inference.queue_timestamp = datetime.now()
    ...

def add_to_blending_queue(event: HairstyleChangeEvent) -> None:
    for blending_event in event.blend_inferences:
        blending_event.queue_timestamp = datetime.now()
    ...
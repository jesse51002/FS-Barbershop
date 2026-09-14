from datetime import datetime, timedelta
import typing
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel

EVENT_TIMEOUT = timedelta(hours=1)

class Hairstyle(BaseModel):
    hairstyle_id: int
    hairstyle_name: str
    
    color_id: int
    color_name: str
    
class UploadPicture(BaseModel):
    file_location: str
    bbox: tuple[int, int, int, int]

class EmbeddingInferenceResult(BaseModel):
    inference_eventid: str
    hairchange_eventid: str
    
    embedded_file_location: str
    segmentation_file_location: str
        
    errored: bool
    
class BlendInferenceResult(BaseModel):
    inference_eventid: str
    hairchange_eventid: str
    
    result_img_location: str
    
    errored: bool
    
class InferenceEvent(BaseModel):
    inference_eventid: str
    
    type: Literal["Embedding", "Blending"]
    
    start_timestamp: Optional[datetime] = None
    
    hairstyle: Optional[Hairstyle] = None
    
    result: Optional[EmbeddingInferenceResult | BlendInferenceResult] = None

    queue_timestamp: Optional[datetime] = None
    finished_timestamp: Optional[datetime] = None
        
    def __init__(self, **data: typing.Any) -> None:
        data["start_timestamp"] = datetime.now()
        super().__init__(**data)
        
    def set_result(self, result: EmbeddingInferenceResult | BlendInferenceResult):
        self.result = result
        self.finished_timestamp = datetime.now()

class HairstyleChangeEvent(BaseModel):
    eventid: str
    account_identifier: str
    
    uploaded_picture: Optional[UploadPicture] = None
    hairstyles: Optional[list[Hairstyle]] = None
    
    embedding_inference: Optional[InferenceEvent] = None
    blend_inferences: Optional[list[InferenceEvent]] = None
    
    start_timestamp: datetime
    
    uploaded_picture_timestamp: Optional[datetime] = None
    picked_hairstyles_timestamp: Optional[datetime] = None
    
    finished_timestamp: Optional[datetime] = None
    
    event_timeout: datetime
    
    errored: bool = False

        
    def __init__(self, **data: typing.Any) -> None:
        data["start_timestamp"] = datetime.now()
        data["event_timeout"] = datetime.now() + EVENT_TIMEOUT
        super().__init__(**data)

def create_eventid() -> str:
    """
    Creates a unique event ID using the current date and time.

    Returns:
        eventid (str): The unique event ID.

    This function generates a unique event identifier using the current date and time,
    and a random UUID. It then returns the event identifier as a string.
    """
    eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
    return eventid

def get_event(eventid: str, account_identifier: Optional[str] = None) -> HairstyleChangeEvent:
    """
    Asserts that the given account identifier has a change event with the specified event ID.

    Args:
        eventid (str): The ID of the event.
        account_identifier (Optional[str]): The identifier of the account.
            If it is supplied, the account must have an event with the specified ID.

    Returns:
        None

    Raises:
        Exception: If the event with the specified ID does not exist.
        Exception: If the account does not have an event with the specified ID.
    """
    event = get_data(eventid)
    
    if event is None:
        raise Exception(f"The event with id {eventid} does not exist.")
    
    if account_identifier is not None and event.account_identifier != account_identifier:
        raise PermissionError(f"Account {account_identifier} does not have an event with id {eventid}.")
    
    if datetime.now() > event.event_timeout:
        handle_timeout(eventid=eventid)
        raise TimeoutError("Event has timed out")
    
    return event

def handle_timeout(eventid: str):
    raise NotImplementedError()

import os
import json

database = "database/"
os.makedirs(database, exist_ok=True)

def write_data(data: HairstyleChangeEvent):
    json_file = os.path.join(database, f"{data.eventid}.json")
    
    with open(json_file, "w") as f:
        f.write(data.model_dump_json())
        
def get_data(eventid: str) -> HairstyleChangeEvent:
    json_file = os.path.join(database,f"{eventid}.json")
    
    if not os.path.exists(json_file):
        return None
    
    with open(json_file, "r") as f:
        return HairstyleChangeEvent(**json.load(f))
    
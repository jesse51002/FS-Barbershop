from datetime import datetime
import typing
from typing import Optional

from hairstyle_creation.errors import AlreadyExists, EmbeddingNotFinished
from hairstyle_creation.models import (
    UploadPicture,
    Hairstyle,
    HairstyleChangeEvent,
    BlendInferenceResult,
    create_eventid,
    write_data,
    get_event
)
from hairstyle_creation.handlers.inference_handler import (
    start_embedding_inference,
    start_blending_inference
)

def create_new_hairstyle_event(
    account_identifier: str,
    ):
    """
    Creates a new hairstyle change event and uploads it to the database.

    Args:
        account_identifier (str): The identifier of the account associated with the event.

    Returns:
        eventid (str): The unique identifier of the newly created event.

    It then creates a new `HairstyleChangeEvent` object with the
    generated event identifier and the provided account identifier.
    """
    eventid = create_eventid()
    
    event = HairstyleChangeEvent(
        eventid = eventid,
        account_identifier = account_identifier
    )
    
    # Uploads the event to the database
    write_data(event)
    
    return eventid


def add_hairstyles(account_identifier: str, eventid: str, hairstyles_dict: list[dict[str, typing.Any]]) -> Optional[Exception]:
    """
    Adds hairstyles choices to a change event in the database.

    Args:
        account_identifier (str): The identifier of the account associated with the change event.
        eventid (str): The unique identifier of the change event.
        hairstyles (list[dict[str, typing.Any]]): A list of dictionaries representing the hairstyles choices.

    Returns:
        None
   
    Raises:
        Exception: If the account does not have an event with the provided eventid.
    """
    event = get_event(eventid, account_identifier)
    
    if event.picked_hairstyles_timestamp is not None:
        return AlreadyExists("Already picked hairstyles for this event")
    
    hairstyles: list[Hairstyle] = []
    
    for style_dict in hairstyles_dict:
        hairstyle = Hairstyle(**style_dict)
        assert hairstyle not in hairstyles
        hairstyles.append(hairstyle)
    
    event.hairstyles = hairstyles
    event.picked_hairstyles_timestamp = datetime.now()
    
    # Starts the blending inference if embedding has finished
    # If embedding has not finished then blending will start automatically when embedding is finished
    try:
        start_blending_inference(event)
    except EmbeddingNotFinished:
        pass
    finally:
        write_data(event)
   
 
def add_uploaded_picture(account_identifier: str, eventid: str, picture: dict[str, typing.Any]) -> Optional[Exception]:
    """
    Adds uploaded pictures to a change event in the database.

    Args:
        account_identifier (str): The identifier of the account associated with the change event.
        eventid (str): The unique identifier of the change event.
        pictures (list[dict[str, typing.Any]]): A list of dictionaries representing the uploaded pictures.

    Returns:
        (UserError | None): Either None if the upload was successful, or a UserError if the upload was unsuccessful.
   
    Raises:
        Exception: If the account does not have an event with the provided eventid.
    """
    event = get_event(eventid, account_identifier)
    
    if event.uploaded_picture_timestamp is not None:
        return AlreadyExists("Already uploaded a picture for this event")
    
    event.uploaded_picture = UploadPicture(**picture)
    event.uploaded_picture_timestamp = datetime.now()
    
    # Starts the embedding event immediately the picture is uploaded to reduce customer waiting time
    start_embedding_inference(event)
    
    write_data(event)
    

def get_results(account_identifier: str, eventid: str) -> list[BlendInferenceResult] | None:
    """
    Get the hair change results for a specific account and event.

    Args:
        account_identifier (str): The identifier of the account associated with the change event.
        eventid (str): The unique identifier of the change event.

    Returns:
        list[BlendInferenceResult] | None: The list of hair inference results for the event, or None if the event is not finished.
    """
    event = get_event(eventid, account_identifier)
    
    # If it has not started, return False
    if (event.start_timestamp is None or
        event.uploaded_picture is None or
        event.hairstyles is None or
        event.embedding_inference is None or 
        event.blend_inferences is None):
        return None
    
    blend_results: list[BlendInferenceResult] = []
    
    if len(event.blend_inferences) != len(event.hairstyles):
        return None
    
    for inference_event in event.blend_inferences:
        # If any inference event has not finished, return None because it is not finished
        if inference_event.result is None:
            return None
        
        blend_results.append(inference_event.result)
    
    # If all inference events have finished, set the event as finished
    if event.finished_timestamp is not None:
        event.finished_timestamp = datetime.now()
    
    return blend_results



from datetime import datetime
import typing
from typing import Optional

from hairstyle_creation.errors import AlreadyExists, EmbeddingNotFinished
from hairstyle_creation.models import (
    HairstyleChangeEvent,
    InferenceEvent,
    EmbeddingInferenceResult,
    BlendInferenceResult,
    create_eventid,
    write_data,
    get_event
)
from hairstyle_creation.handlers.aws_queue_handler import add_to_embedding_queue, add_to_blending_queue


def start_embedding_inference(event: HairstyleChangeEvent) -> Optional[Exception]:
    """
    Starts the embedding inference process for a given account and event.

    Args:
        account_identifier (str): The identifier of the account.
        eventid (str): The identifier of the event.

    Returns:
        Optional[Exception]: If the inference process has already started for the event, returns a AlreadyExists

    Raises:
        KeyError: If there is no uploaded picture to embed

    """
        
    # If embedding has already started
    if event.embedding_inference is not None:
        raise AlreadyExists("Embedding has already started")
    
    if event.uploaded_picture is None:
        raise KeyError("There is no uploaded picture to Embed")
    
    inference_eventid = create_eventid()
    inference_event = InferenceEvent(
        inference_eventid = inference_eventid,
        type = "Embedding",
    )
    
    event.embedding_inference = inference_event
    add_to_embedding_queue(event)
        
    
def post_embed_result(result: dict[str, typing.Any]) -> None:
    """
    Update the inference event with the embedding results and start the blending inference if the event has already started.

    Args:
        result (dict[str, typing.Any]): A dictionary containing the embedding results.

    Returns:
        None
    """
    embedding_results = EmbeddingInferenceResult(**result)
    
    event = get_event(embedding_results.hairchange_eventid)
    
    if event.embedding_inference is None:
        raise KeyError("Embedding has not started")
    
    if event.embedding_inference.inference_eventid != embedding_results.inference_eventid:
        raise KeyError("The hairchange event does not match the embedding inference event")
    
    if event.embedding_inference.result is not None:
        raise AlreadyExists("Embedding results have already been posted")
    
    event.embedding_inference.set_result(embedding_results)
    
    try:
        # When embedding is finished trys to start blending
        # If embedding finished before user has picked hairstyles then user will start blending when they pick hairstyles
        start_blending_inference(event)
    except KeyError as e:
        pass
    finally:
        write_data(event)


def start_blending_inference(event: HairstyleChangeEvent) -> Optional[Exception]:
    """
    Start the blending inference process for a given inference event ID.
    
    Args:
        hairchange_eventid (str): The ID of the inference event.
        emdedding_results (EmbeddingInferenceResult): The embedding results for the inference event.
    
    Raises:
        KeyError: If the hairstyles to blend have not been picked by user yet
    
    Returns:
        starting_inference (bool): Whether the inference process was started or not. (True if started, False if not started())
        or
        AlreadyExists: If the inference process has already started for the hairchange event.
    """
    
    # Can not start if the hairstyles have not been picked yet
    if event.hairstyles is None:
        raise KeyError("There are no hairstyles to blend")
    
    if event.embedding_inference is None:
        raise ValueError("Embedding has not startet yet")
        
    if event.embedding_inference.result is None:
        raise EmbeddingNotFinished("Embedding has not finished yet")
    
    if event.blend_inferences is None:
        event.blend_inferences = []
    
    # Makes sure that the blending process has not already started
    for blend_inference_event in event.blend_inferences:
        blend_hairstyle = blend_inference_event.hairstyle
        for hairstyle in event.hairstyles:
            if blend_hairstyle == hairstyle:
                raise AlreadyExists("Already started Blending")
        
    for hairstyle in event.hairstyles:
        inference_eventid = create_eventid()
        inference_event = InferenceEvent(
            inference_eventid = inference_eventid,
            type = "Blending",
            hairstyle = hairstyle
        )
        event.blend_inferences.append(inference_event)
    
    add_to_blending_queue(event)        
    

def post_blend_result(result: dict[str, typing.Any]):
    """
    Update the blending result for a given inference event.

    Args:
        result (dict[str, typing.Any]): A dictionary containing the blending result.

    Raises:
        AssertionError: If the inference event ID is not found in the inference database.
    """
    blending_results = BlendInferenceResult(**result)
    
    event = get_event(blending_results.hairchange_eventid)
    
    if event.blend_inferences is None:
        raise KeyError("This hairchange event does not match the blending inference event")
    
    for blend_inference_event in event.blend_inferences:
        if blend_inference_event.inference_eventid != blending_results.inference_eventid:
            continue
        
        if blend_inference_event.result is not None:
            raise AlreadyExists("Blending result have already been posted")
        
        blend_inference_event.set_result(blending_results)
        
        # Checks if all blending inferences are finished
        all_done = True
        for blend_inference_event in event.blend_inferences:
            if blend_inference_event.result is None:
                all_done = False
                break
        
        if all_done:
            event.finished_timestamp = datetime.now()
        
        write_data(event)
        return
        
    raise KeyError("This hairchange event does not match the blending inference event")
    

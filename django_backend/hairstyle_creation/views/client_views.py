from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

from datetime import datetime
from uuid import uuid4
import json

from hairstyle_creation.handlers.client_event_handler import (
    create_new_hairstyle_event,
    add_hairstyles,
    get_results,
    add_uploaded_picture
    )

from hairstyle_creation.handlers.inference_handler import (
    start_embedding_inference
    )

# Links for the hairstyles
HAIRSTYLE_PRESET_LINKS = [
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQk0WZTFFVUB6WVoVzpJUL1e9CnUKcC4NFFQ&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQqw-QW4mIv2iUX4lGZlhbX0fh644yRVfwEwA&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTNhll_-c0syMENDjudM9K-KFiitxIZ4iumg&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRs5mHctzXpwLoaD7DMpXaMg67hh-9zmf-BQw&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcShUzrAMs9I-qU5p_SZgWRoQYno7M57WDWEiKhZWiLbqtst8_L4KmeXHB_j5LQ36HcIT8A&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRoZxffMAns0_2l7DBCcIt-KG7NTCMRx7KxKg&usqp=CAU"
    ]


custom_hairstyle_events = {}

hairstyle_rendering_events = {}

# This is called at the start of the hairstyle creation in order to get the Creation ID
# Input: None
# Output: EventID
def start_creation(request):
    if request.method != "GET":
        return HttpResponseBadRequest("Must use a GET request")
    
    # Starts the event
    eventid = create_new_hairstyle_event(
        account_identifier=""
    )
    
    return JsonResponse({"eventid": eventid})


# This supplies the image links and id for all the hairstyle presets
# Input: EventID
# Output: List of Hairstyles IDs and links
def get_hairstyles_presets(request):
    if request.method != "GET":
        return HttpResponseBadRequest("Must use a GET request")

    eventid = request.GET.get('eventid')
    print("EventID: ",eventid)
    # Valides eventID
    if eventid == None:
        return HttpResponseBadRequest("Invalid or missing EventID")
    
    # Creates a list of links and ids
    hairstyles = []
    for i, link in enumerate(HAIRSTYLE_PRESET_LINKS):
        hairstyles.append({
            "hairstlye_id":i, 
            "hairstyle_url": link
            }
            )
    
    # Returns data
    return JsonResponse({"Hairstyles": hairstyles})

@csrf_exempt 
def start_rendering(request):

    if request.method != "POST":
        return HttpResponseBadRequest("Must use a POST request")
    
    eventid = request.GET.get('eventid')
    print("EventID: ",eventid)
    # Valides eventID
    if eventid == None:
        return HttpResponseBadRequest("Invalid or missing EventID")
    
    body = json.loads(request.body)
    
    hairstyles = body["hairstyles"]
    
    add_hairstyles(account_identifier="", eventid=eventid, hairstyles=hairstyles)
    start_embedding_inference(account_identifier="", eventid=eventid)

    # Returns data
    return JsonResponse({"sucess": True})

@csrf_exempt 
def add_uploaded_picture_request(request):

    if request.method != "POST":
        return HttpResponseBadRequest("Must use a POST request")
    
    eventid = request.GET.get('eventid')
    print("EventID: ",eventid)
    # Valides eventID
    if eventid == None:
        return HttpResponseBadRequest("Invalid or missing EventID")
    
    body = json.loads(request.body)
    
    picture = body["photo_link"]
    
    add_uploaded_picture(account_identifier="", eventid=eventid, picture=picture)
    start_embedding_inference(account_identifier="", eventid=eventid)
    
    # Returns data
    return JsonResponse({"sucess": True})

# This returns the image transformation results or its status
# Will return true after being called 3 or more times
# Input: EventID
# Output: custom image id and in_progress boolean
def get_rendering_results(request):
    if request.method != "GET":
        return HttpResponseBadRequest("Must use a GET request")
    
    eventid = request.GET.get('eventid')
    print("EventID: ",eventid)
    # Valides eventID
    if eventid == None:
        return HttpResponseBadRequest("Invalid or missing EventID")
    
    # if the event doesnt exist then invalid request
    if eventid not in hairstyle_rendering_events:
        return HttpResponseBadRequest("Event ID does not have a rendering event")
    
    results = get_results(account_identifier="", eventid=eventid)
    
    if isinstance(results, Exception):
        return 
    
    # Returns data
    return JsonResponse({"results": results})
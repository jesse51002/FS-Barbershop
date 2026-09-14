from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

from datetime import datetime
from uuid import uuid4
import json

from hairstyle_creation.handlers.inference_handler import (
    post_blend_result, 
    post_embed_result
    )

"""
Its ok to send full errors here because it is going securly to AWS
"""

@csrf_exempt 
def blend_results_request(request):

    if request.method != "POST":
        return HttpResponseBadRequest("Must use a POST request")
    
    eventid = request.GET.get('eventid')
    print("EventID: ",eventid)
    # Valides eventID
    if eventid == None:
        return HttpResponseBadRequest("Invalid or missing EventID")
    
    body = json.loads(request.body)
    
    post_blend_result(body)
    
    # Returns data
    return JsonResponse({"sucess": True})

@csrf_exempt 
def embedding_results_request(request):

    if request.method != "POST":
        return HttpResponseBadRequest("Must use a POST request")
    
    eventid = request.GET.get('eventid')
    print("EventID: ",eventid)
    # Valides eventID
    if eventid == None:
        return HttpResponseBadRequest("Invalid or missing EventID")
    
    body = json.loads(request.body)
    
    post_embed_result(body)
    
    # Returns data
    return JsonResponse({"sucess": True})
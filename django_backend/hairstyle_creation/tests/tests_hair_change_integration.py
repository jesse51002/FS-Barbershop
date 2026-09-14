from datetime import datetime

from hairstyle_creation.errors import AlreadyExists
from hairstyle_creation.models import (
    BlendInferenceResult,
    EmbeddingInferenceResult,
    HairstyleChangeEvent,
    Hairstyle,
    UploadPicture,
    InferenceEvent,
    get_event,
    write_data,
    create_eventid
)
from hairstyle_creation.handlers.inference_handler import (
    start_embedding_inference,
    start_blending_inference,
    post_blend_result,
    post_embed_result
)
from hairstyle_creation.handlers.client_event_handler import (
    create_new_hairstyle_event,
    add_hairstyles,
    add_uploaded_picture,
    get_results
)

from hairstyle_creation.tests.test_presets import (
    blend_inference_valid,
    blend_inference_result_valid,
    picture_valid,
    embedding_inference_valid,
    embedding_inference_result_valid,
    hairstyle_1,
    hairstyle_2
)

from django.test import TestCase

ACCOUNT_IDENTIFIER = "test"


class HairChangeIntergrationTest(TestCase):
    def setUp(self):
        self.event_id = create_new_hairstyle_event(account_identifier=ACCOUNT_IDENTIFIER)

    def test_intergration_pick_hair_first(self):
        add_uploaded_picture(account_identifier=ACCOUNT_IDENTIFIER, eventid=self.event_id, picture=picture_valid)
        add_hairstyles(account_identifier=ACCOUNT_IDENTIFIER, eventid=self.event_id, hairstyles_dict=[hairstyle_1, hairstyle_2])
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        
        embed_results = embedding_inference_result_valid.copy()
        embed_results["inference_eventid"] = event.embedding_inference.inference_eventid
        embed_results["hairchange_eventid"] = event.eventid
        post_embed_result(embed_results)
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        
        for blend_inference in event.blend_inferences:
            blend_results = blend_inference_result_valid.copy()
            blend_results["inference_eventid"] = blend_inference.inference_eventid
            blend_results["hairchange_eventid"] = event.eventid            
            post_blend_result(blend_results)
            
        results = get_results(account_identifier=ACCOUNT_IDENTIFIER, eventid=self.event_id)    
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, BlendInferenceResult)

    def test_intergration_embed_finish_first(self):
        add_uploaded_picture(account_identifier=ACCOUNT_IDENTIFIER, eventid=self.event_id, picture=picture_valid)
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        
        embed_results = embedding_inference_result_valid.copy()
        embed_results["inference_eventid"] = event.embedding_inference.inference_eventid
        embed_results["hairchange_eventid"] = event.eventid
        post_embed_result(embed_results)
        
        add_hairstyles(account_identifier=ACCOUNT_IDENTIFIER, eventid=self.event_id, hairstyles_dict=[hairstyle_1, hairstyle_2])
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        
        for blend_inference in event.blend_inferences:
            blend_results = blend_inference_result_valid.copy()
            blend_results["inference_eventid"] = blend_inference.inference_eventid
            blend_results["hairchange_eventid"] = event.eventid            
            post_blend_result(blend_results)
            
        results = get_results(account_identifier=ACCOUNT_IDENTIFIER, eventid=self.event_id)
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, BlendInferenceResult)
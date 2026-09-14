from datetime import datetime

from hairstyle_creation.errors import AlreadyExists, EmbeddingNotFinished
from hairstyle_creation.models import (
    BlendInferenceResult,
    EmbeddingInferenceResult,
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
from hairstyle_creation.handlers.client_event_handler import create_new_hairstyle_event

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


class StartEmbeddingInferenceTest(TestCase):
    def setUp(self):
        self.event_id = create_new_hairstyle_event(account_identifier=ACCOUNT_IDENTIFIER)

    def test_start_embedding_inference_valid(self):
        """Testing embedding inference with a valid picture"""  
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        event.uploaded_picture = UploadPicture(**picture_valid)
        event.uploaded_picture_timestamp = datetime.now()
        write_data(event)
        
        start_embedding_inference(event=event)
        
        self.assertIsInstance(event.embedding_inference, InferenceEvent)
        self.assertIsInstance(event.embedding_inference.start_timestamp, datetime)
        
    def test_start_embedding_inference_already_started(self):
        """Tests embedding inference with an already started inference"""  
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        event.uploaded_picture = UploadPicture(**picture_valid)
        event.uploaded_picture_timestamp = datetime.now()
        event.embedding_inference = InferenceEvent(**embedding_inference_valid)
        write_data(event)
        
        def err_func():
            start_embedding_inference(event) 
        self.assertRaises(AlreadyExists, err_func)       
        
    def test_start_embedding_inference_no_picture(self):
        """Tests embedding inference with an already started inference"""  
        
        event = get_event(eventid=self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        def err_func():
            start_embedding_inference(event)
        self.assertRaises(KeyError, err_func)



class PostEmbeddingResultTest(TestCase):
    def setUp(self):
        self.event_id = create_new_hairstyle_event(account_identifier=ACCOUNT_IDENTIFIER)

    def test_post_embedding_result_valid(self):
        """Testing embedding inference with a valid picture"""  
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        event.uploaded_picture = UploadPicture(**picture_valid)
        event.uploaded_picture_timestamp = datetime.now()
        
        event.embedding_inference = InferenceEvent(**embedding_inference_valid)
        event.embedding_inference.start_timestamp = datetime.now()
        # event.embedding_inference.result = InferenceEvent(**embedding_inference_result_valid)
        write_data(event)
        
        results = embedding_inference_result_valid.copy()
        results["inference_eventid"] = event.embedding_inference.inference_eventid
        results["hairchange_eventid"] = event.eventid
        
        post_embed_result(results)
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        
        self.assertIsInstance(event.embedding_inference.result, EmbeddingInferenceResult)
        self.assertIsInstance(event.embedding_inference.finished_timestamp, datetime)
    
    def test_start_embedding_inference_already_posted(self):
        """Tests embedding inference with an already posted result"""  
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        event.uploaded_picture = UploadPicture(**picture_valid)
        event.uploaded_picture_timestamp = datetime.now()
        
        event.embedding_inference = InferenceEvent(**embedding_inference_valid)
        event.embedding_inference.start_timestamp = datetime.now()
        
        results = embedding_inference_result_valid.copy()
        results["inference_eventid"] = event.embedding_inference.inference_eventid
        results["hairchange_eventid"] = event.eventid
        
        event.embedding_inference.result = EmbeddingInferenceResult(**results)
        write_data(event)
        
        def err_func():
            post_embed_result(results)
        
        self.assertRaises(AlreadyExists, err_func)
        

class StartBlendingInferenceTest(TestCase):
    def setUp(self):
        self.event_id = create_new_hairstyle_event(account_identifier=ACCOUNT_IDENTIFIER)

    def add_requirements(self):
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        event.uploaded_picture = UploadPicture(**picture_valid)
        event.uploaded_picture_timestamp = datetime.now()
        
        event.embedding_inference = InferenceEvent(**embedding_inference_valid)
        event.embedding_inference.start_timestamp = datetime.now()
        
        results = embedding_inference_result_valid.copy()
        results["inference_eventid"] = event.embedding_inference.inference_eventid
        results["hairchange_eventid"] = event.eventid
        
        event.hairstyles = [Hairstyle(**hairstyle_1), Hairstyle(**hairstyle_2)]
        event.picked_hairstyles_timestamp = datetime.now()
        
        event.embedding_inference.result = EmbeddingInferenceResult(**results)
        write_data(event)
        
        return event
    
    def test_start_blending_inference_valid(self):
        """Testing embedding inference with a valid picture"""  
        event = self.add_requirements()
        
        start_blending_inference(event=event)
                
        self.assertIsInstance(event.blend_inferences, list)
        for inference in event.blend_inferences:
            self.assertIsInstance(inference, InferenceEvent)
            self.assertIsInstance(inference.start_timestamp, datetime)
        
    def test_start_blending_inference_already_started(self):
        """Tests blending inference with an already started inference"""  
        event = self.add_requirements()
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        
        event.blend_inferences = [InferenceEvent(**blend_inference_valid) for _ in range(len(event.hairstyles))]
        for i in range(len(event.hairstyles)):
            event.blend_inferences[i].start_timestamp = datetime.now()
            event.blend_inferences[i].hairstyle = event.hairstyles[i]
        
        write_data(event)
        
        def error_func():
            start_blending_inference(event)
        self.assertRaises(AlreadyExists, error_func)
        
    def test_start_blending_inference_no_hairstyles(self):
        """Tests embedding with no hairstyles"""  
        
        event = get_event(eventid=self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        def err_func():
            start_blending_inference(event)
        self.assertRaises(KeyError, err_func)
        
    def test_blending_inference_embeding_not_started(self):
        """Tests embedding inference with an already started inference"""  
        event = self.add_requirements()
        event = get_event(eventid=self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        event.embedding_inference.result = None
        event.embedding_inference.finished_timestamp = None
        def err_func():
            start_blending_inference(event)
        self.assertRaises(EmbeddingNotFinished, err_func)



class PostBlendingResultTest(TestCase):
    def setUp(self):
        self.event_id = create_new_hairstyle_event(account_identifier=ACCOUNT_IDENTIFIER)
    
    def test_post_blending_result_valid(self):
        """Testing posting blending with a valid result"""  
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        event.hairstyles = [Hairstyle(**hairstyle_1), Hairstyle(**hairstyle_2)]
        event.blend_inferences = []
        for _ in range(len(event.hairstyles)):
            infernece_event = InferenceEvent(**blend_inference_valid) 
            infernece_event.inference_eventid = create_eventid()
            event.blend_inferences.append(infernece_event)
        write_data(event)
        
        # Test only the first inference
        # This is to test that finished_timestamp is updated when all inferences are finished
        results = blend_inference_result_valid
        results["inference_eventid"] = event.blend_inferences[0].inference_eventid
        results["hairchange_eventid"] = event.eventid
        post_blend_result(results)
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        self.assertIsNone(event.finished_timestamp)
        
        results = blend_inference_result_valid.copy()
        results["inference_eventid"] = event.blend_inferences[1].inference_eventid
        results["hairchange_eventid"] = event.eventid
        post_blend_result(results)
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        
        self.assertIsInstance(event.finished_timestamp, datetime)
        
        for inference in event.blend_inferences:
            self.assertIsInstance(inference.finished_timestamp, datetime)
            self.assertIsInstance(inference.result, BlendInferenceResult)
        
    
    def test_start_blending_inference_already_posted(self):
        """Tests embedding inference with an already started inference"""  
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        event.hairstyles = [Hairstyle(**hairstyle_1), Hairstyle(**hairstyle_2)]
        event.blend_inferences = []
        for _ in range(len(event.hairstyles)):
            infernece_event = InferenceEvent(**blend_inference_valid) 
            infernece_event.inference_eventid = create_eventid()
            infernece_event.result = BlendInferenceResult(**blend_inference_result_valid)
            event.blend_inferences.append(infernece_event)
        write_data(event)
        
        def error_func():
            results = blend_inference_result_valid.copy()
            results["inference_eventid"] = event.blend_inferences[0].inference_eventid
            results["hairchange_eventid"] = event.eventid
            post_blend_result(results)
        self.assertRaises(AlreadyExists, error_func)
        
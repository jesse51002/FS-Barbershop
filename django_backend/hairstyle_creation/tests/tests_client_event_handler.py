from datetime import datetime

import pydantic_core
from hairstyle_creation.errors import AlreadyExists
from hairstyle_creation.models import (
    BlendInferenceResult,
    HairstyleChangeEvent,
    Hairstyle,
    UploadPicture,
    InferenceEvent,
    get_event,
    write_data
)
from hairstyle_creation.handlers.client_event_handler import (
    create_new_hairstyle_event,
    add_hairstyles,
    add_uploaded_picture,
    get_results
)

from hairstyle_creation.tests.test_presets import (
    blend_inference_result_valid,
    picture_valid,
    picture_invalid,
    hairstyle_1,
    hairstyle_2,
    hairstyle_invalid,
    embedding_inference_valid
)

from django.test import TestCase

ACCOUNT_IDENTIFIER = "test"



# Create your tests here.
class CreateEventTest(TestCase):
    def setUp(self):
        ...

    def test_create_event(self):
        """Tests that the event is created correctly"""
        event_id = create_new_hairstyle_event(account_identifier=ACCOUNT_IDENTIFIER)
        
        self.assertRegex(event_id, "^[0-9]{6}-[0-9]{4}-[0-9]{4}-[0-9a-f-]{36}$")
        
        event = get_event(event_id, account_identifier=ACCOUNT_IDENTIFIER)
        
        self.assertIsInstance(event, HairstyleChangeEvent)
        self.assertEqual(event.account_identifier, ACCOUNT_IDENTIFIER)
        self.assertEqual(event.eventid, event_id)
        self.assertIsInstance(event.event_timeout, datetime)
        self.assertIsInstance(event.start_timestamp, datetime)

    
class AddHairstyleTest(TestCase):
    def setUp(self):
        self.event_id = create_new_hairstyle_event(account_identifier=ACCOUNT_IDENTIFIER)

    def test_add_hairstyle_valid(self):
        """Tests adding a valid hairstyle"""
        
        hairstyles_dict = [hairstyle_1, hairstyle_2]
        
        # Writes embedding inference because embedding should've started before user picks hairstyles
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        event.embedding_inference = InferenceEvent(
            inference_eventid="oijasd",
            type="Embedding"
        )
        write_data(event)
        
        add_hairstyles(
            account_identifier=ACCOUNT_IDENTIFIER,
            eventid=self.event_id,
            hairstyles_dict=hairstyles_dict,
            )
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        
        self.assertEqual(len(event.hairstyles), len(hairstyles_dict))
        
        for hairstyle in hairstyles_dict:
            hairstyle = Hairstyle(**hairstyle)
            self.assertIn(hairstyle, event.hairstyles)
            
    def test_add_hairstyle_duplicate(self):
        """Tests adding a duplicate hairstyle"""
        
        hairstyles_dict = [hairstyle_1, hairstyle_1]
        
        def duplicate_run():
            add_hairstyles(
                account_identifier=ACCOUNT_IDENTIFIER,
                eventid=self.event_id,
                hairstyles_dict=hairstyles_dict,
            )
        
        self.assertRaises(Exception, duplicate_run)
        
    def test_add_invalid_hairstyle(self):
        """Tests invalid hairstyles dict"""
        
        hairstyles_dict = [hairstyle_invalid]
        
        def invlaid_dict():
            add_hairstyles(
                account_identifier=ACCOUNT_IDENTIFIER,
                eventid=self.event_id,
                hairstyles_dict=hairstyles_dict,
            )
        
        self.assertRaises(pydantic_core._pydantic_core.ValidationError, invlaid_dict)
    
    def test_hairstyles_already_picked(self):
        """Tests adding hairstyles for an already picked event"""
        
        hairstyles_dict = [hairstyle_1, hairstyle_2]
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        event.picked_hairstyles_timestamp = datetime.now()
        event.hairstyles = [Hairstyle(**hairstyle) for hairstyle in hairstyles_dict]
        write_data(event)
        
        add_execption = add_hairstyles(
            account_identifier=ACCOUNT_IDENTIFIER,
            eventid=self.event_id,
            hairstyles_dict=hairstyles_dict,
        )
        
        self.assertIsInstance(add_execption, AlreadyExists)
        
        
        
class AddPictureTest(TestCase):
    
    
    def setUp(self):
        self.event_id = create_new_hairstyle_event(account_identifier=ACCOUNT_IDENTIFIER)

    def test_add_picture_valid(self):
        """Tests adding a valid hairstyle"""  
        add_uploaded_picture(
            account_identifier=ACCOUNT_IDENTIFIER,
            eventid=self.event_id,
            picture=picture_valid,
        )
        
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        
        self.assertEqual(event.uploaded_picture, UploadPicture(**picture_valid))
        self.assertIsInstance(event.uploaded_picture_timestamp, datetime)
        self.assertIsInstance(event.embedding_inference, InferenceEvent)
        
    def test_add_invalid_picture(self):
        """Tests invalid hairstyles dict"""        
        def invlaid_dict():
            add_uploaded_picture(
                account_identifier=ACCOUNT_IDENTIFIER,
                eventid=self.event_id,
                picture=picture_invalid,
            )
        
        self.assertRaises(pydantic_core._pydantic_core.ValidationError, invlaid_dict)
    
    def test_picture_already_added(self):
        """Tests adding hairstyles for an already picked event"""
                
        event = get_event(self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        event.uploaded_picture_timestamp = datetime.now()
        event.uploaded_picture = UploadPicture(**picture_valid)
        write_data(event)
        
        add_execption = add_uploaded_picture(
            account_identifier=ACCOUNT_IDENTIFIER,
            eventid=self.event_id,
            picture=picture_valid,
        )
        
        self.assertIsInstance(add_execption, AlreadyExists)
        
        
class GetResultsTest(TestCase):
    finished_event = HairstyleChangeEvent(
        account_identifier=ACCOUNT_IDENTIFIER,
        eventid="oijasd",
        event_timeout=datetime.now(),
        start_timestamp=datetime.now(),
        picked_hairstyles_timestamp=datetime.now(),
        uploaded_picture_timestamp=datetime.now(),
        uploaded_picture=UploadPicture(**picture_valid),
        embedding_inference=InferenceEvent(**embedding_inference_valid),
        hairstyles=[Hairstyle(**hairstyle_1)],
        blend_inferences=[
            InferenceEvent(
                inference_eventid="oijasd",
                type="Blending",
                finished_timestamp=datetime.now(),
                errored=False,
                result=BlendInferenceResult(**blend_inference_result_valid)
            ) 
        ],
        finished_timestamp=datetime.now()
    )
    
    def setUp(self):
        self.event_id = create_new_hairstyle_event(account_identifier=ACCOUNT_IDENTIFIER)

    def test_get_results_finished(self):
        """Tests getting results on a finished event"""  
        event = self.finished_event
        event.eventid = self.event_id
        write_data(event)
        
        results = get_results(eventid=self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, BlendInferenceResult)
        
    def test_get_results_not_finished(self):
        """Tests invalid hairstyles dict"""        
        results = get_results(eventid=self.event_id, account_identifier=ACCOUNT_IDENTIFIER)
        self.assertIsNone(results)

from typing import List, Optional
from uagents import Model

# Match your FastAPI slide shape
class Slide(Model):
    content: str
    image_url: Optional[str] = None

# Tutor is bound to one course, so request is empty
class GetCourseRequest(Model):
    pass

class GetCourseResponse(Model):
    slides: List[Slide]

# Match your /answer_base64 JSON response
class AnswerStepRequest(Model):
    session_id: str
    image_b64: str            # required (your API requires an image each call)

class AnswerStepResponse(Model):
    answer_value: str
    solving_completed: bool

# Minimal probe for listing/labeling tutors
class TutorInfoRequest(Model):
    pass

class TutorInfoResponse(Model):
    tutor_id: str  # e.g., "GEOMETRY"

class ErrorMessage(Model):
    error: str
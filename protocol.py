from uagents import Protocol
from models import (
    GetCourseRequest, GetCourseResponse,
    AnswerStepRequest, AnswerStepResponse,
    TutorInfoRequest, TutorInfoResponse,
)

# bump when you change any model/replies
tutor_proto_v1 = Protocol(name="StudySnapsTutorProtocol", version="1.0.7")
# IMPORTANT: No @on_message handlers here.

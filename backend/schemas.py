from pydantic import BaseModel
from typing import List

class EvaluationInput(BaseModel):
    age: int
    gender: str
    stream: str
    internships: int
    cgpa: float
    history_of_backlogs: int
    role: str
    skills: List[str]

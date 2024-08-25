
p_b = "7ddpd33ZiN6khFjGmEg-uA%3D%3D"
p_lat = "YFSLJUdtEgZhSTP3teHpKvWaS8Df8AN936BF1Fa0zQ%3D%3D"

os.environ["MAGENTIC_BACKEND"] = "poe_api_wrapper"
os.environ["MAGENTIC_POE_API_WRAPPER_TOKEN_P_B"] = p_b
os.environ["MAGENTIC_POE_API_WRAPPER_TOKEN_P_LAT"] = p_lat

from magentic import prompt
from pydantic import BaseModel
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional

####### FRAGMENTS SECTION ##################

class Vigilance(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    arterial_tension: Optional[str]
    cardiac_frequency: Optional[str]
    dejections: Optional[str]
    diary_interpretation_id: Optional[int] = Field(default=None, foreign_key="diaryinterpretation.id")

class Interocurrence(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_type: str
    event_description: str
    event_date: str
    diary_interpretation_id: Optional[int] = Field(default=None, foreign_key="diaryinterpretation.id")

class Medication(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    medication_name: str
    dosage: str
    administration_time: str
    diary_interpretation_id: Optional[int] = Field(default=None, foreign_key="diaryinterpretation.id")

class Observation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    observation_type: str
    observation_details: str
    diary_interpretation_id: Optional[int] = Field(default=None, foreign_key="diaryinterpretation.id")

####### END OF FRAGMENTS SECTION ##################

class DiaryInterpretation(BaseModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    vigilances: Vigilance = Relationship(back_populates="diary_interpretation")
    interocurrences: List[Interocurrence] = Relationship(back_populates="diary_interpretation")
    medications: List[Medication] = Relationship(back_populates="diary_interpretation")
    other_observations: List[Observation] = Relationship(back_populates="diary_interpretation")

Vigilance.diary_interpretation = Relationship(back_populates="vigilances")
Interocurrence.diary_interpretation = Relationship(back_populates="interocurrences")
Medication.diary_interpretation = Relationship(back_populates="medications")
Observation.diary_interpretation = Relationship(back_populates="other_observations")


from magentic import prompt
from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
from magentic.chat_model.litellm_chat_model import LitellmChatModel
import json

# Local imports
from MODELS.clinical_diaries_parametrization import DiaryInterpretation
from load_configs import *

@prompt("""
    Clinical diary:
    {diary_text}

    Extract the parameterized information. Don't return anything else like 'here is the patient's information:' """)
def interpret_diary(diary_text: str) -> DiaryInterpretation: ...

test = interpret_diary("here is the diary lmao")

print(test)

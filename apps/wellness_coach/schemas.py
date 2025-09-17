# from typing import List, Literal, Annotated
# from pydantic import BaseModel, Field

# class SuggestionItem(BaseModel):
#     title: str = Field(max_length=80)
#     rationale: str = Field(max_length=300)
#     steps: Annotated[List[str], Field(min_length=1)]

# class WellnessOutput(BaseModel):
#     summary: str = Field(max_length=400)
#     suggestions: Annotated[List[SuggestionItem], Field(min_length=2)]
#     disclaimer: Literal["This is general wellness info, not medical advice."]

from pydantic import BaseModel, Field, constr
from typing import List

# Accept any disclaimer that clearly says "not medical advice" (case-insensitive)
DisclaimerType = constr(strip_whitespace=True, pattern=r"(?i).*not medical advice.*")

Short = constr(strip_whitespace=True, min_length=3, max_length=80)
Medium = constr(strip_whitespace=True, min_length=10, max_length=400)
Summary = constr(strip_whitespace=True, min_length=30, max_length=500)

class SuggestionItem(BaseModel):
    title: Short
    rationale: Medium
    # Allow at least 1, up to 6 steps; each step short
    steps: List[Short] = Field(min_length=1, max_length=6)

class WellnessOutput(BaseModel):
    summary: Summary
    # Allow 1–5 suggestions (we’d LIKE 2+, but don’t fail the whole thing if it returns 1)
    suggestions: List[SuggestionItem] = Field(min_length=1, max_length=5)
    disclaimer: DisclaimerType

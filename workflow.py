import json
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

class StudyPlan(BaseModel):
    target_audience: str = Field(description="Target skill level (e.g., Beginner, Intermediate, Advanced)")
    core_modules: list[str] = Field(description="3-5 key learning subtopics or modules")
    learning_objectives: list[str] = Field(description="Concrete learning outcomes")

class ReviewVerdict(BaseModel):
    is_valid: bool = Field(description="True if generated content meets all criteria and plan requirements")
    quality_score: int = Field(description="Quality score from 1 to 10")
    feedback: str = Field(description="Detailed feedback or suggested improvements")

class StudyPackPipeline:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        # Primary and fallback models for high-traffic spikes
        self.primary_model = "gemini-2.5-flash"
        self.fallback_model = "gemini-2.5-pro"

    def _call_with_retry(self, **kwargs):
        """
        Executes API calls with automatic exponential backoff retries.
        Falls back to an alternative model if 503 errors persist.
        """
        max_retries = 3
        backoff_delay = 2  # Delay in seconds

        # Try primary model first, then fallback model
        for current_model in [self.primary_model, self.fallback_model]:
            kwargs["model"] = current_model
            for attempt in range(max_retries):
                try:
                    return self.client.models.generate_content(**kwargs)
                except APIError as err:
                    if getattr(err, "code", None) == 503 or "503" in str(err):
                        if attempt < max_retries - 1:
                            time.sleep(backoff_delay * (2 ** attempt))
                            continue
                    # If retries fail on primary model, break loop to switch to fallback
                    break
                except Exception as err:
                    raise err

        # Final attempt fallback trigger
        return self.client.models.generate_content(**kwargs)

    def stage_plan(self, topic: str, goal: str, prompts: dict) -> StudyPlan:
        prompt = prompts["planner"].format(topic=topic, goal=goal)
        response = self._call_with_retry(
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StudyPlan,
            ),
        )
        return StudyPlan.model_validate_json(response.text)

    def stage_generate(self, topic: str, plan: StudyPlan, prompts: dict) -> str:
        prompt = prompts["generator"].format(
            topic=topic,
            audience=plan.target_audience,
            modules=", ".join(plan.core_modules),
            objectives=", ".join(plan.learning_objectives)
        )
        response = self._call_with_retry(contents=prompt)
        return response.text

    def stage_review(self, plan: StudyPlan, content: str, prompts: dict) -> ReviewVerdict:
        prompt = prompts["reviewer"].format(
            plan=plan.model_dump_json(),
            content=content
        )
        response = self._call_with_retry(
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReviewVerdict,
            ),
        )
        return ReviewVerdict.model_validate_json(response.text)

    def stage_refine(self, content: str, feedback: str, prompts: dict) -> str:
        prompt = prompts["refiner"].format(
            feedback=feedback,
            content=content
        )
        response = self._call_with_retry(contents=prompt)
        return response.text

    def run_pipeline(self, topic: str, goal: str, prompts: dict, status_callback) -> tuple[StudyPlan, str, ReviewVerdict]:
        status_callback("Stage 1/4: Planning curriculum blueprint...")
        plan = self.stage_plan(topic, goal, prompts)

        status_callback("Stage 2/4: Generating modules, flashcards, and assessment...")
        raw_content = self.stage_generate(topic, plan, prompts)

        status_callback("Stage 3/4: Auditing content for quality and plan adherence...")
        review = self.stage_review(plan, raw_content, prompts)

        status_callback("Stage 4/4: Polishing final study pack layout...")
        final_content = self.stage_refine(raw_content, review.feedback, prompts)

        return plan, final_content, review
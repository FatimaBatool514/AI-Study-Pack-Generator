import json
from google import genai
from google.genai import types
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
        self.model = "gemini-3.6-flash"

    def stage_plan(self, topic: str, goal: str, prompts: dict) -> StudyPlan:
        """Stage 1: Planning - Generates structured curriculum blueprint using Pydantic schema."""
        prompt = prompts["planner"].format(topic=topic, goal=goal)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StudyPlan,
            ),
        )
        return StudyPlan.model_validate_json(response.text)

    def stage_generate(self, topic: str, plan: StudyPlan, prompts: dict) -> str:
        """Stage 2: Content Generation - Creates comprehensive explanations, flashcards, and quiz."""
        prompt = prompts["generator"].format(
            topic=topic,
            audience=plan.target_audience,
            modules=", ".join(plan.core_modules),
            objectives=", ".join(plan.learning_objectives)
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text

    def stage_review(self, plan: StudyPlan, content: str, prompts: dict) -> ReviewVerdict:
        """Stage 3: Review & Audit - Validates content quality and compliance against plan."""
        prompt = prompts["reviewer"].format(
            plan=plan.model_dump_json(),
            content=content
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReviewVerdict,
            ),
        )
        return ReviewVerdict.model_validate_json(response.text)

    def stage_refine(self, content: str, feedback: str, prompts: dict) -> str:
        """Stage 4: Refinement & Formatting - Refines content based on audit feedback."""
        prompt = prompts["refiner"].format(
            feedback=feedback,
            content=content
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text

    def run_pipeline(self, topic: str, goal: str, prompts: dict, status_callback) -> tuple[StudyPlan, str, ReviewVerdict]:
        """Orchestrates multi-stage AI workflow execution."""
        status_callback("Stage 1/4: Planning curriculum blueprint...")
        plan = self.stage_plan(topic, goal, prompts)

        status_callback("Stage 2/4: Generating modules, flashcards, and assessment...")
        raw_content = self.stage_generate(topic, plan, prompts)

        status_callback("Stage 3/4: Auditing content for quality and plan adherence...")
        review = self.stage_review(plan, raw_content, prompts)

        status_callback("Stage 4/4: Polishing final study pack layout...")
        final_content = self.stage_refine(raw_content, review.feedback, prompts)

        return plan, final_content, review

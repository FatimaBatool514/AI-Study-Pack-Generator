import json
import re
from typing import Any, Dict, List

from utils import call_ai, call_ai_json


def planning_stage(
    client,
    topic: str,
    material: str,
    level: str,
    goals: str,
    study_days: int,
    model: str,
) -> Dict[str, Any]:
    """Stage 1: create a personalized learning plan."""
    system = """
You are an expert instructional designer and curriculum planner.
Analyze the learner's topic, level, goals, source material, and study time.
Create a precise plan for a personalized study pack.
Do not generate the final study material yet.
"""

    prompt = f"""
Topic: {topic}
Student level: {level}
Learning goals: {goals or "Understand the topic and prepare for assessment."}
Available study days: {study_days}

Source material:
{material or "No source material was provided. Use reliable general knowledge."}

Return ONLY JSON:
{{
  "learning_objectives": ["..."],
  "subtopics": ["..."],
  "difficulty": "beginner/intermediate/advanced",
  "priority_topics": ["..."],
  "content_strategy": "...",
  "assessment_strategy": "...",
  "study_schedule_strategy": "..."
}}
"""
    result = call_ai_json(client, system, prompt, model, temperature=0.2)
    result["stage"] = "Planning"
    return result


def content_generation_stage(
    client,
    plan: Dict[str, Any],
    topic: str,
    material: str,
    level: str,
    mcq_count: int,
    flashcard_count: int,
    model: str,
) -> Dict[str, Any]:
    """Stage 2: generate study content using planning context."""
    system = """
You are an expert teacher and educational content writer.
Create accurate, clear, student-friendly material.
Use the planning context from Stage 1 and prioritize supplied source material.
Avoid unsupported or fabricated facts.
"""

    prompt = f"""
Topic: {topic}
Student level: {level}

STAGE 1 PLANNING CONTEXT:
{json.dumps(plan, indent=2, ensure_ascii=False)}

SOURCE MATERIAL:
{material or "No source material was provided."}

Return ONLY JSON:
{{
  "overview": "...",
  "detailed_notes": "...",
  "quick_revision": ["..."],
  "key_terms": [
    {{"term": "...", "definition": "..."}}
  ],
  "examples": ["..."],
  "mcqs": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "explanation": "..."
    }}
  ],
  "flashcards": [
    {{"question": "...", "answer": "..."}}
  ],
  "short_questions": [
    {{"question": "...", "answer": "..."}}
  ]
}}

Requirements:
- Exactly {mcq_count} MCQs.
- Exactly {flashcard_count} flashcards.
- Exactly 5 short-answer questions.
- Every MCQ has A, B, C and D.
- Avoid duplicates.
- Keep difficulty aligned with the planning stage.
"""
    result = call_ai_json(client, system, prompt, model, temperature=0.5)
    result["stage"] = "Content Generation"
    return result


def assessment_stage(
    client,
    plan: Dict[str, Any],
    content: Dict[str, Any],
    mcq_count: int,
    flashcard_count: int,
    model: str,
) -> Dict[str, Any]:
    """Stage 3: evaluate generated content."""
    system = """
You are a rigorous educational QA specialist.
Check generated content for accuracy, completeness, duplication, ambiguity,
answer consistency, coverage, and alignment with the learner's level.
"""

    prompt = f"""
STAGE 1 PLAN:
{json.dumps(plan, indent=2, ensure_ascii=False)}

STAGE 2 CONTENT:
{json.dumps(content, indent=2, ensure_ascii=False)}

Expected MCQs: {mcq_count}
Expected flashcards: {flashcard_count}

Return ONLY JSON:
{{
  "overall_score": 0,
  "pass": true,
  "checks": {{
    "coverage": {{"score": 0, "issues": ["..."]}},
    "accuracy": {{"score": 0, "issues": ["..."]}},
    "mcq_quality": {{"score": 0, "issues": ["..."]}},
    "flashcard_quality": {{"score": 0, "issues": ["..."]}},
    "level_alignment": {{"score": 0, "issues": ["..."]}},
    "clarity": {{"score": 0, "issues": ["..."]}}
  }},
  "critical_issues": ["..."],
  "minor_issues": ["..."],
  "required_changes": ["..."]
}}

Use scores from 0 to 100.
Set pass=false when important corrections are required.
"""
    result = call_ai_json(client, system, prompt, model, temperature=0.1)
    result["stage"] = "Assessment / QA"
    return result


def review_stage(
    client,
    plan: Dict[str, Any],
    content: Dict[str, Any],
    assessment: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """Stage 4: decide what needs refinement."""
    system = """
You are a senior instructional reviewer.
Use the plan, generated content, and QA report to decide whether the pack
should be approved or refined. Identify concrete changes.
"""

    prompt = f"""
PLAN:
{json.dumps(plan, indent=2, ensure_ascii=False)}

CONTENT:
{json.dumps(content, indent=2, ensure_ascii=False)}

ASSESSMENT:
{json.dumps(assessment, indent=2, ensure_ascii=False)}

Return ONLY JSON:
{{
  "decision": "approve" or "refine",
  "review_summary": "...",
  "strengths": ["..."],
  "changes": [
    {{
      "area": "...",
      "problem": "...",
      "action": "..."
    }}
  ]
}}
"""
    result = call_ai_json(client, system, prompt, model, temperature=0.1)
    result["stage"] = "Review"
    return result


def refinement_stage(
    client,
    plan: Dict[str, Any],
    content: Dict[str, Any],
    assessment: Dict[str, Any],
    review: Dict[str, Any],
    topic: str,
    level: str,
    mcq_count: int,
    flashcard_count: int,
    model: str,
) -> Dict[str, Any]:
    """Stage 5: produce the final improved study pack."""
    system = """
You are the final educational editor.
Refine the generated study pack using all previous workflow context.
Preserve correct content and fix identified problems.
Do not introduce unsupported facts.
"""

    prompt = f"""
Topic: {topic}
Student level: {level}

PLAN:
{json.dumps(plan, indent=2, ensure_ascii=False)}

ORIGINAL CONTENT:
{json.dumps(content, indent=2, ensure_ascii=False)}

ASSESSMENT:
{json.dumps(assessment, indent=2, ensure_ascii=False)}

REVIEW:
{json.dumps(review, indent=2, ensure_ascii=False)}

Return ONLY JSON:
{{
  "overview": "...",
  "detailed_notes": "...",
  "quick_revision": ["..."],
  "key_terms": [
    {{"term": "...", "definition": "..."}}
  ],
  "examples": ["..."],
  "mcqs": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "explanation": "..."
    }}
  ],
  "flashcards": [
    {{"question": "...", "answer": "..."}}
  ],
  "short_questions": [
    {{"question": "...", "answer": "..."}}
  ],
  "study_plan": ["Day 1: ...", "Day 2: ..."],
  "final_quality_note": "..."
}}

Keep exactly {mcq_count} MCQs and {flashcard_count} flashcards.
Keep exactly 5 short-answer questions.
"""
    result = call_ai_json(client, system, prompt, model, temperature=0.3)
    result["stage"] = "Refinement"
    return result


def validate_final_content(
    content: Dict[str, Any],
    expected_mcqs: int,
    expected_flashcards: int,
) -> List[str]:
    """Deterministic validation after the AI refinement stage."""
    errors = []

    required = [
        "overview",
        "detailed_notes",
        "quick_revision",
        "key_terms",
        "mcqs",
        "flashcards",
        "short_questions",
    ]

    for key in required:
        if key not in content:
            errors.append(f"Missing field: {key}")

    mcqs = content.get("mcqs", [])
    flashcards = content.get("flashcards", [])
    short_questions = content.get("short_questions", [])

    if len(mcqs) != expected_mcqs:
        errors.append(
            f"Expected {expected_mcqs} MCQs, received {len(mcqs)}."
        )

    if len(flashcards) != expected_flashcards:
        errors.append(
            f"Expected {expected_flashcards} flashcards, received {len(flashcards)}."
        )

    if len(short_questions) != 5:
        errors.append(
            f"Expected 5 short-answer questions, received {len(short_questions)}."
        )

    seen_questions = set()

    for index, mcq in enumerate(mcqs, start=1):
        if not all(
            key in mcq for key in ["question", "options", "answer", "explanation"]
        ):
            errors.append(f"MCQ {index} is missing required fields.")
            continue

        options = mcq.get("options", {})
        if set(options.keys()) != {"A", "B", "C", "D"}:
            errors.append(f"MCQ {index} must have A/B/C/D options.")

        if mcq.get("answer") not in {"A", "B", "C", "D"}:
            errors.append(f"MCQ {index} has an invalid answer.")

        normalized = re.sub(r"\s+", " ", mcq["question"].lower()).strip()
        if normalized in seen_questions:
            errors.append(f"Duplicate MCQ detected at question {index}.")
        seen_questions.add(normalized)

    return errors


def run_full_workflow(
    client,
    topic: str,
    material: str,
    level: str,
    goals: str,
    study_days: int,
    mcq_count: int,
    flashcard_count: int,
    model: str,
) -> Dict[str, Any]:
    """
    Main workflow controller.

    Context flow:
    Planning → Content → Assessment → Review → Refinement
    """
    plan = planning_stage(
        client, topic, material, level, goals, study_days, model
    )

    content = content_generation_stage(
        client,
        plan,
        topic,
        material,
        level,
        mcq_count,
        flashcard_count,
        model,
    )

    assessment = assessment_stage(
        client,
        plan,
        content,
        mcq_count,
        flashcard_count,
        model,
    )

    review = review_stage(
        client,
        plan,
        content,
        assessment,
        model,
    )

    final_content = refinement_stage(
        client,
        plan,
        content,
        assessment,
        review,
        topic,
        level,
        mcq_count,
        flashcard_count,
        model,
    )

    validation_errors = validate_final_content(
        final_content,
        mcq_count,
        flashcard_count,
    )

    return {
        "planning": plan,
        "content_generation": content,
        "assessment": assessment,
        "review": review,
        "refinement": final_content,
        "validation_errors": validation_errors,
    }

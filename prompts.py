import json
import os
import re
from typing import Any, Dict

from openai import OpenAI


def get_client(api_key: str = ""):
    """Create the OpenAI client from user input or environment variable."""
    key = api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()

    if not key:
        return None

    return OpenAI(api_key=key)


def call_ai(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.3,
) -> str:
    """Centralized AI call with basic error handling."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
    except Exception as exc:
        raise RuntimeError(f"AI API request failed: {exc}") from exc

    if not response.choices:
        raise RuntimeError("AI returned no choices.")

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("AI returned empty content.")

    return content.strip()


def call_ai_json(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Call AI and safely parse a JSON object."""
    raw = call_ai(
        client,
        system_prompt
        + "\nYou MUST return a single valid JSON object. No Markdown fences.",
        user_prompt,
        model,
        temperature,
    )

    cleaned = raw.strip()

    # Recover JSON if the model accidentally uses ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end <= start:
            raise RuntimeError(
                "AI returned invalid JSON. Please run the workflow again."
            )

        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "AI returned malformed JSON. Please run the workflow again."
            ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("AI JSON response must be an object.")

    return data


def build_markdown(topic: str, level: str, workflow: Dict[str, Any]) -> str:
    """Convert the final workflow output into a downloadable Markdown file."""
    final = workflow["refinement"]
    assessment = workflow["assessment"]

    lines = [
        f"# 📚 AI Study Pack — {topic}",
        "",
        f"**Student Level:** {level}",
        "",
        "## 1. Topic Overview",
        final.get("overview", ""),
        "",
        "## 2. Detailed Study Notes",
        final.get("detailed_notes", ""),
        "",
        "## 3. Quick Revision Notes",
    ]

    for item in final.get("quick_revision", []):
        lines.append(f"- {item}")

    lines += ["", "## 4. Key Terms"]

    for item in final.get("key_terms", []):
        lines.append(
            f"**{item.get('term', '')}:** {item.get('definition', '')}"
        )

    lines += ["", "## 5. Examples"]

    for item in final.get("examples", []):
        lines.append(f"- {item}")

    lines += ["", "## 6. MCQs"]

    for i, mcq in enumerate(final.get("mcqs", []), start=1):
        lines.append(f"### MCQ {i}")
        lines.append(mcq.get("question", ""))

        for letter in ["A", "B", "C", "D"]:
            lines.append(
                f"- **{letter}.** {mcq.get('options', {}).get(letter, '')}"
            )

        lines.append(f"**Answer:** {mcq.get('answer', '')}")
        lines.append(f"**Explanation:** {mcq.get('explanation', '')}")
        lines.append("")

    lines += ["## 7. Flashcards"]

    for i, card in enumerate(final.get("flashcards", []), start=1):
        lines.append(f"### Flashcard {i}")
        lines.append(f"**Q:** {card.get('question', '')}")
        lines.append(f"**A:** {card.get('answer', '')}")
        lines.append("")

    lines += ["## 8. Short Answer Questions"]

    for i, question in enumerate(final.get("short_questions", []), start=1):
        lines.append(f"### Question {i}")
        lines.append(question.get("question", ""))
        lines.append(f"**Model Answer:** {question.get('answer', '')}")
        lines.append("")

    lines += ["## 9. Study Plan"]

    for item in final.get("study_plan", []):
        lines.append(f"- {item}")

    lines += [
        "",
        "## 10. Quality Review",
        f"**AI Quality Score:** {assessment.get('overall_score', 'N/A')}/100",
        f"**QA Status:** {'PASS' if assessment.get('pass') else 'REVIEW'}",
        "",
        final.get("final_quality_note", ""),
    ]

    return "\n".join(lines)

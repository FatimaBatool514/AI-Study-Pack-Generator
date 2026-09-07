# prompts.py
PLANNER_PROMPT = """You are an expert curriculum designer and educational consultant.
Design a structured study plan blueprint for the requested topic and goal.

Topic: {topic}
Learning Goal: {goal}

Analyze the target complexity and return a structured JSON response matching the required schema.
"""

GENERATOR_PROMPT = """You are a senior domain expert and educator.
Create a comprehensive, high-quality study pack based strictly on the provided plan context.

Topic: {topic}
Target Audience: {audience}
Core Modules: {modules}
Learning Objectives: {objectives}

Requirements:
1. Theoretical Explanation: In-depth, clear explanation for each module.
2. Flashcards: 5 high-yield Question & Answer pairs covering key concepts.
3. Assessment: A 3-question conceptual quiz with step-by-step answer keys and explanations.
"""

REVIEWER_PROMPT = """You are an academic auditor and quality control expert.
Evaluate the generated study material against the original curriculum plan.

Target Curriculum Plan:
{plan}

Generated Study Content:
{content}

Perform a rigorous check for accuracy, module completeness, and audience suitability.
Return your evaluation according to the required verdict schema.
"""

REFINER_PROMPT = """You are an expert technical editor and instructional designer.
Refine and polish the study pack content based on auditor feedback.

Audit Feedback:
{feedback}

Original Content:
{content}

Formatting Rules:
- Ensure clean Markdown syntax with clear headings.
- Use bullet points, bold emphasis for key terminology, and clear quiz keys.
"""

PROMPTS = {
    "planner": PLANNER_PROMPT,
    "generator": GENERATOR_PROMPT,
    "reviewer": REVIEWER_PROMPT,
    "refiner": REFINER_PROMPT
}
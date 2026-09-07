import os
import re
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import faiss
import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# AI STUDY PACK WORKFLOW
# Plan -> Retrieve -> Generate -> Assess -> Review -> Refine
# ============================================================

st.set_page_config(
    page_title="AI Study Pack Workflow",
    page_icon="🧠",
    layout="wide",
)

DEFAULT_LLM = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class WorkflowContext:
    document_name: str = ""
    student_level: str = "Undergraduate"
    learning_goal: str = ""
    focus_topic: str = ""
    study_time: int = 60
    difficulty: str = "Balanced"
    flashcards: int = 10
    mcqs: int = 10

    raw_text: str = ""
    chunks: List[str] = None
    plan: Dict[str, Any] = None
    retrieved_context: str = ""
    draft_pack: str = ""
    assessment: Dict[str, Any] = None
    review: Dict[str, Any] = None
    final_pack: str = ""

    def __post_init__(self):
        self.chunks = self.chunks or []
        self.plan = self.plan or {}
        self.assessment = self.assessment or {}
        self.review = self.review or {}


# -----------------------------
# Cached models / clients
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_groq_client(api_key: str):
    return Groq(api_key=api_key)


# -----------------------------
# Utilities
# -----------------------------
def extract_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_no, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"[Page {page_no}]\n{text}")

    doc.close()
    return "\n\n".join(pages)


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 220) -> List[str]:
    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def build_index(chunks: List[str], model):
    vectors = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(query: str, chunks: List[str], index, model, top_k: int = 6):
    if not chunks or index is None:
        return []

    q = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, ids = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            results.append((float(score), chunks[int(idx)]))

    return results


def format_context(results: List[Tuple[float, str]]) -> str:
    return "\n\n---\n\n".join(
        f"Source {i + 1} (similarity={score:.3f}):\n{chunk}"
        for i, (score, chunk) in enumerate(results)
    )


def safe_json(text: str) -> Dict[str, Any]:
    """Parse model JSON while tolerating markdown code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return {"raw": text, "parse_error": True}


def llm_json(client, model, system_prompt, user_prompt) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0.15,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return safe_json(response.choices[0].message.content)


def llm_text(client, model, system_prompt, user_prompt, temperature=0.25) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ============================================================
# STAGE 1 — PLANNING AGENT
# ============================================================
def planning_stage(client, model, ctx: WorkflowContext) -> Dict[str, Any]:
    system = """You are the Planning Agent in an educational AI workflow.
Create a personalized study-pack plan from the learner profile and document
metadata. Do not invent document facts. Return valid JSON only."""

    prompt = f"""
Learner level: {ctx.student_level}
Learning goal: {ctx.learning_goal or "General mastery"}
Focus topic: {ctx.focus_topic or "Main topics in the document"}
Available study time: {ctx.study_time} minutes
Difficulty preference: {ctx.difficulty}
Requested flashcards: {ctx.flashcards}
Requested MCQs: {ctx.mcqs}
Document: {ctx.document_name}
Document chunk count: {len(ctx.chunks)}

Create JSON with:
{{
  "learning_objectives": ["..."],
  "priority_topics": ["..."],
  "recommended_sequence": ["..."],
  "content_mix": {{
      "summary": true,
      "key_concepts": true,
      "flashcards": {ctx.flashcards},
      "mcqs": {ctx.mcqs},
      "self_test": 5
  }},
  "difficulty_strategy": "...",
  "quality_criteria": ["grounded", "clear", "accurate", "useful"]
}}
"""
    return llm_json(client, model, system, prompt)


# ============================================================
# STAGE 2 — CONTEXT / RETRIEVAL AGENT
# ============================================================
def retrieval_stage(ctx: WorkflowContext, index, embedding_model, top_k=7):
    query_parts = [
        ctx.focus_topic,
        ctx.learning_goal,
        "definitions concepts explanations examples formulas important exam material",
        " ".join(ctx.plan.get("priority_topics", [])),
    ]
    query = " ".join(p for p in query_parts if p)

    results = retrieve(query, ctx.chunks, index, embedding_model, top_k)
    ctx.retrieved_context = format_context(results)
    return results


# ============================================================
# STAGE 3 — CONTENT GENERATION AGENT
# ============================================================
def generation_stage(client, model, ctx: WorkflowContext) -> str:
    system = """You are the Content Generation Agent.
Generate a high-quality study pack grounded ONLY in the supplied retrieved
document context. Never fabricate facts. Follow the planning context.
Use Markdown. Make it useful for active recall and exam preparation."""

    prompt = f"""
PLAN:
{json.dumps(ctx.plan, indent=2)}

LEARNER:
Level: {ctx.student_level}
Goal: {ctx.learning_goal}
Focus: {ctx.focus_topic}
Difficulty: {ctx.difficulty}
Study time: {ctx.study_time} minutes

RETRIEVED DOCUMENT CONTEXT:
{ctx.retrieved_context}

Create:

# Study Pack

## 1. Executive Summary
Concise and grounded.

## 2. Learning Objectives
Based on the plan.

## 3. Key Concepts
Important concepts with explanations.

## 4. Flashcards
Create exactly {ctx.flashcards}.

Format:
**Q1:** ...
**A1:** ...

## 5. Multiple Choice Questions
Create exactly {ctx.mcqs}.
Each must have A-D options, correct answer, and a brief explanation.

## 6. Exam-Focused Review
Important relationships, processes, formulas, distinctions, or facts.

## 7. Quick Self-Test
Create exactly 5 questions without answers.

Every factual claim must be supported by the retrieved material.
"""
    return llm_text(client, model, system, prompt)


# ============================================================
# STAGE 4 — ASSESSMENT AGENT
# ============================================================
def assessment_stage(client, model, ctx: WorkflowContext) -> Dict[str, Any]:
    system = """You are the Assessment Agent.
Audit a generated study pack for educational quality and grounding.
Return valid JSON only. Be strict."""

    prompt = f"""
ORIGINAL PLAN:
{json.dumps(ctx.plan, indent=2)}

RETRIEVED CONTEXT:
{ctx.retrieved_context}

DRAFT STUDY PACK:
{ctx.draft_pack}

Evaluate:
- factual grounding
- completeness against the plan
- flashcard count
- MCQ count
- MCQ correctness
- ambiguity
- hallucination risk
- level appropriateness
- pedagogical usefulness

Return:
{{
  "overall_score": 0,
  "grounding_score": 0,
  "completeness_score": 0,
  "assessment_quality_score": 0,
  "issues": ["..."],
  "required_fixes": ["..."],
  "pass": true
}}

Scores are 0-100.
Set pass=false if material correction is required.
"""
    result = llm_json(client, model, system, prompt)

    # Defensive validation
    result.setdefault("overall_score", 0)
    result.setdefault("issues", [])
    result.setdefault("required_fixes", [])
    result["pass"] = bool(result.get("pass", False))
    return result


# ============================================================
# STAGE 5 — REVIEW AGENT
# ============================================================
def review_stage(client, model, ctx: WorkflowContext) -> Dict[str, Any]:
    system = """You are the Review Agent.
Review the draft and assessment report. Identify concrete corrections needed.
Do not rewrite the entire study pack. Return valid JSON only."""

    prompt = f"""
PLAN:
{json.dumps(ctx.plan, indent=2)}

ASSESSMENT:
{json.dumps(ctx.assessment, indent=2)}

DRAFT:
{ctx.draft_pack}

RETRIEVED CONTEXT:
{ctx.retrieved_context}

Return:
{{
  "decision": "refine",
  "strengths": ["..."],
  "critical_corrections": ["..."],
  "minor_corrections": ["..."],
  "preserve": ["..."]
}}

Decision must be one of: "approve", "refine", "regenerate".
"""
    return llm_json(client, model, system, prompt)


# ============================================================
# STAGE 6 — REFINEMENT AGENT
# ============================================================
def refinement_stage(client, model, ctx: WorkflowContext) -> str:
    system = """You are the Refinement Agent.
Produce the final study pack by correcting the draft according to the
assessment and review. Preserve correct content. Use ONLY the supplied
retrieved context for factual claims. Return Markdown only."""

    prompt = f"""
PLAN:
{json.dumps(ctx.plan, indent=2)}

ASSESSMENT:
{json.dumps(ctx.assessment, indent=2)}

REVIEW:
{json.dumps(ctx.review, indent=2)}

RETRIEVED CONTEXT:
{ctx.retrieved_context}

DRAFT:
{ctx.draft_pack}

Create the final corrected study pack.

Requirements:
- Exactly {ctx.flashcards} flashcards.
- Exactly {ctx.mcqs} MCQs.
- Exactly 5 self-test questions.
- Correct errors identified by assessment/review.
- Remove unsupported claims.
- Keep the student's level ({ctx.student_level}) in mind.
- Keep the requested focus ({ctx.focus_topic or "main document topics"}).
- Make the final result clear and exam-ready.
"""
    return llm_text(client, model, system, prompt, temperature=0.2)


# ============================================================
# WORKFLOW ORCHESTRATOR + ERROR HANDLING
# ============================================================
def run_workflow(ctx, client, model, index, embedding_model, top_k=7):
    stages = []

    try:
        stages.append(("1. Planning", "running"))
        ctx.plan = planning_stage(client, model, ctx)
        stages[-1] = ("1. Planning", "complete")

        stages.append(("2. Retrieval", "running"))
        results = retrieval_stage(ctx, index, embedding_model, top_k)
        if not results:
            raise RuntimeError("No relevant document chunks were retrieved.")
        stages[-1] = ("2. Retrieval", "complete")

        stages.append(("3. Content Generation", "running"))
        ctx.draft_pack = generation_stage(client, model, ctx)
        if not ctx.draft_pack:
            raise RuntimeError("The content-generation stage returned empty output.")
        stages[-1] = ("3. Content Generation", "complete")

        stages.append(("4. Assessment", "running"))
        ctx.assessment = assessment_stage(client, model, ctx)
        stages[-1] = ("4. Assessment", "complete")

        stages.append(("5. Review", "running"))
        ctx.review = review_stage(client, model, ctx)
        stages[-1] = ("5. Review", "complete")

        decision = ctx.review.get("decision", "refine")

        stages.append(("6. Refinement", "running"))
        if decision == "approve" and ctx.assessment.get("pass", False):
            ctx.final_pack = ctx.draft_pack
        else:
            ctx.final_pack = refinement_stage(client, model, ctx)

        if not ctx.final_pack:
            raise RuntimeError("The refinement stage returned empty output.")

        stages[-1] = ("6. Refinement", "complete")

        return ctx, stages, None

    except Exception as exc:
        if stages:
            current = stages[-1][0]
            stages[-1] = (current, f"error: {exc}")
        return ctx, stages, str(exc)


# ============================================================
# STREAMLIT STATE
# ============================================================
if "ctx" not in st.session_state:
    st.session_state.ctx = WorkflowContext()

if "index" not in st.session_state:
    st.session_state.index = None

if "workflow_log" not in st.session_state:
    st.session_state.workflow_log = []

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = ""


# ============================================================
# UI
# ============================================================
st.title("🧠 AI Study Pack Generator")
st.caption(
    "Multi-stage AI workflow: Planning → Retrieval → Generation → Assessment → Review → Refinement"
)

with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
    )

    llm_model = st.text_input("Groq model", value=DEFAULT_LLM)

    student_level = st.selectbox(
        "Student level",
        ["Beginner", "High School", "Undergraduate", "Graduate", "Professional"],
        index=2,
    )

    learning_goal = st.text_area(
        "Learning goal",
        placeholder="e.g. Prepare for my final exam and understand the core concepts.",
    )

    focus_topic = st.text_input(
        "Focus topic",
        placeholder="e.g. Chapter 3 / Thermodynamics",
    )

    study_time = st.slider("Study time (minutes)", 15, 240, 60, step=15)

    difficulty = st.select_slider(
        "Difficulty",
        options=["Easy", "Balanced", "Challenging"],
        value="Balanced",
    )

    flashcards = st.slider("Flashcards", 5, 30, 10)
    mcqs = st.slider("MCQs", 5, 25, 10)
    top_k = st.slider("Retrieved chunks", 3, 10, 7)


uploaded = st.file_uploader("📄 Upload your study PDF", type=["pdf"])

if uploaded:
    if st.session_state.indexed_file != uploaded.name:
        try:
            with st.spinner("Extracting and indexing PDF..."):
                raw_text = extract_pdf(uploaded.getvalue())

                if not raw_text.strip():
                    st.error(
                        "This PDF has no selectable text. Add OCR support for scanned PDFs."
                    )
                    st.stop()

                chunks = chunk_text(raw_text)
                embedding_model = load_embedding_model()
                index = build_index(chunks, embedding_model)

                st.session_state.ctx = WorkflowContext(
                    document_name=uploaded.name,
                    student_level=student_level,
                    learning_goal=learning_goal,
                    focus_topic=focus_topic,
                    study_time=study_time,
                    difficulty=difficulty,
                    flashcards=flashcards,
                    mcqs=mcqs,
                    raw_text=raw_text,
                    chunks=chunks,
                )
                st.session_state.index = index
                st.session_state.indexed_file = uploaded.name
                st.session_state.workflow_log = []
                st.session_state.qa_history = []

            st.success(
                f"Indexed {uploaded.name}: {len(raw_text):,} characters, "
                f"{len(chunks)} chunks."
            )
        except Exception as exc:
            st.error(f"PDF processing failed: {exc}")

if st.session_state.index is not None:
    # Sync current learner controls into context
    ctx = st.session_state.ctx
    ctx.student_level = student_level
    ctx.learning_goal = learning_goal
    ctx.focus_topic = focus_topic
    ctx.study_time = study_time
    ctx.difficulty = difficulty
    ctx.flashcards = flashcards
    ctx.mcqs = mcqs

    st.success(
        f"Knowledge base ready: **{ctx.document_name}** · "
        f"{len(ctx.chunks)} chunks"
    )

    stage_cols = st.columns(6)
    stage_names = [
        "Planning",
        "Retrieval",
        "Generation",
        "Assessment",
        "Review",
        "Refinement",
    ]

    for col, name in zip(stage_cols, stage_names):
        with col:
            st.markdown(f"**{name}**")

    if st.button(
        "🚀 Run Full AI Workflow",
        type="primary",
        use_container_width=True,
    ):
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
        else:
            try:
                client = load_groq_client(api_key)

                with st.status("Running multi-stage AI workflow...", expanded=True) as status:
                    updated_ctx, log, error = run_workflow(
                        ctx,
                        client,
                        llm_model,
                        st.session_state.index,
                        load_embedding_model(),
                        top_k,
                    )

                    st.session_state.ctx = updated_ctx
                    st.session_state.workflow_log = log

                    for stage, state in log:
                        if state == "complete":
                            st.write(f"✅ {stage}")
                        elif state.startswith("error"):
                            st.write(f"❌ {stage}: {state}")

                    if error:
                        status.update(
                            label="Workflow stopped with an error",
                            state="error",
                            expanded=True,
                        )
                    else:
                        status.update(
                            label="Workflow completed successfully",
                            state="complete",
                            expanded=False,
                        )

            except Exception as exc:
                st.error(f"Workflow initialization failed: {exc}")

    # Workflow transparency
    if st.session_state.workflow_log:
        with st.expander("🔍 Workflow execution log", expanded=False):
            for stage, state in st.session_state.workflow_log:
                st.write(f"**{stage}:** {state}")

    ctx = st.session_state.ctx

    # Pipeline context inspection
    with st.expander("🧩 Shared workflow context", expanded=False):
        st.json({
            "document": ctx.document_name,
            "student_level": ctx.student_level,
            "learning_goal": ctx.learning_goal,
            "focus_topic": ctx.focus_topic,
            "study_time": ctx.study_time,
            "difficulty": ctx.difficulty,
            "chunk_count": len(ctx.chunks),
            "plan": ctx.plan,
            "assessment": ctx.assessment,
            "review": ctx.review,
        })

    if ctx.plan:
        st.subheader("🗺️ Stage 1 — Personalized Plan")
        st.json(ctx.plan)

    if ctx.assessment:
        st.subheader("📊 Stage 4 — Assessment")
        score = ctx.assessment.get("overall_score", 0)
        st.metric("Overall quality score", f"{score}/100")

        if ctx.assessment.get("issues"):
            st.warning("\n".join(f"• {x}" for x in ctx.assessment["issues"]))

    if ctx.review:
        st.subheader("🔎 Stage 5 — Review")
        st.write(f"Decision: **{ctx.review.get('decision', 'unknown')}**")

        corrections = ctx.review.get("critical_corrections", [])
        if corrections:
            st.warning("\n".join(f"• {x}" for x in corrections))

    if ctx.final_pack:
        st.divider()
        st.subheader("📚 Final Refined Study Pack")
        st.markdown(ctx.final_pack)

        st.download_button(
            "⬇️ Download Study Pack",
            data=ctx.final_pack,
            file_name="ai_study_pack.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.divider()
        st.subheader("💬 AI Tutor — Ask About the PDF")

        question = st.text_input(
            "Question",
            placeholder="What is the most important concept in this chapter?",
        )

        if st.button("🔎 Ask AI Tutor", use_container_width=True):
            if not api_key:
                st.error("Enter your Groq API key.")
            elif not question.strip():
                st.warning("Enter a question.")
            else:
                try:
                    client = load_groq_client(api_key)
                    results = retrieve(
                        question,
                        ctx.chunks,
                        st.session_state.index,
                        load_embedding_model(),
                        top_k,
                    )

                    context = format_context(results)

                    answer = llm_text(
                        client,
                        llm_model,
                        """You are a grounded AI tutor. Answer only from the
retrieved document context. If the context does not support the answer,
say so clearly. Keep explanations student-friendly.""",
                        f"QUESTION:\n{question}\n\nCONTEXT:\n{context}",
                        temperature=0.2,
                    )

                    st.session_state.qa_history.append(
                        {"question": question, "answer": answer}
                    )

                except Exception as exc:
                    st.error(f"AI Tutor failed: {exc}")

        for item in reversed(st.session_state.qa_history):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.markdown("---")

else:
    st.markdown(
        """
## How the AI workflow works

```text
PDF
 │
 ▼
Document Processing
 │
 ▼
┌──────────────────────┐
│ 1. PLANNING AGENT    │
│ Learner goals        │
│ Level + time         │
│ Difficulty + focus   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. RETRIEVAL         │
│ Embeddings + FAISS   │
│ Relevant PDF context │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. GENERATION AGENT  │
│ Summary              │
│ Concepts             │
│ Flashcards           │
│ MCQs                 │
│ Self-test            │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. ASSESSMENT AGENT  │
│ Grounding            │
│ Accuracy             │
│ Completeness         │
│ Assessment quality   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 5. REVIEW AGENT      │
│ Approve / Refine /   │
│ Regenerate           │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 6. REFINEMENT AGENT  │
│ Apply corrections    │
│ Produce final pack   │
└──────────┬───────────┘
           ▼
      FINAL STUDY PACK
```

### Context passing

Every stage receives the shared `WorkflowContext`, so the workflow can pass:

- learner profile
- learning goal
- document information
- retrieved evidence
- planning decisions
- generated draft
- assessment results
- reviewer corrections

### Error handling

The orchestrator stops safely when:

- PDF extraction fails
- no document text exists
- no relevant chunks are retrieved
- an LLM stage returns empty output
- a workflow stage raises an exception

The Streamlit interface exposes the workflow execution log so you can see which stage failed.
"""
    )

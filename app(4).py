import json
import re

import streamlit as st

from utils import build_markdown, get_client
from workflow import run_full_workflow


# ============================================================
# STREAMLIT MAIN APPLICATION
# ============================================================

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Study Pack Generator")
st.write(
    "A multi-stage AI workflow that plans, generates, assesses, reviews, "
    "and refines personalized study material."
)

st.info(
    "Workflow: ① Planning → ② Content Generation → "
    "③ Assessment → ④ Review → ⑤ Refinement"
)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="For deployment, preferably store this as a secret.",
    )

    model = st.text_input(
        "AI Model",
        value="gpt-4.1-mini",
    )

    level = st.selectbox(
        "Student Level",
        [
            "School",
            "College",
            "University",
            "Competitive Exam",
        ],
        index=2,
    )

    study_days = st.slider(
        "Available Study Days",
        min_value=1,
        max_value=30,
        value=7,
    )

    mcq_count = st.slider(
        "Number of MCQs",
        min_value=5,
        max_value=30,
        value=10,
    )

    flashcard_count = st.slider(
        "Number of Flashcards",
        min_value=5,
        max_value=30,
        value=10,
    )


# ------------------------------------------------------------
# Input section
# ------------------------------------------------------------

st.subheader("1️⃣ Learner Profile & Input")

col1, col2 = st.columns(2)

with col1:
    topic = st.text_input(
        "Study Topic",
        placeholder="Example: Project Risk Management",
    )

with col2:
    goals = st.text_input(
        "Learning Goal",
        placeholder="Example: Prepare for an exam and understand key concepts",
    )

material = st.text_area(
    "Study Material / Notes",
    height=240,
    placeholder=(
        "Paste your lecture notes, textbook content, syllabus, or other "
        "study material here..."
    ),
)

uploaded_file = st.file_uploader(
    "Optional: Upload TXT notes",
    type=["txt"],
)

if uploaded_file is not None:
    try:
        uploaded_text = uploaded_file.read().decode("utf-8")
        material = (material + "\n\n" + uploaded_text).strip()
        st.success("TXT file loaded successfully.")
    except Exception as exc:
        st.error(f"Could not read the TXT file: {exc}")


# ------------------------------------------------------------
# Run workflow
# ------------------------------------------------------------

st.divider()

run_button = st.button(
    "🚀 Run Complete AI Workflow",
    type="primary",
    use_container_width=True,
)

if run_button:
    if not topic.strip():
        st.error("Please enter a study topic.")
        st.stop()

    client = get_client(api_key)

    if client is None:
        st.error(
            "OpenAI API key not found. Enter it in the sidebar or configure "
            "OPENAI_API_KEY in your deployment secrets."
        )
        st.stop()

    try:
        progress = st.progress(0)
        status = st.empty()

        status.info("Stage 1/5: Planning learner objectives...")
        progress.progress(10)

        # The workflow controller handles all five stages.
        workflow = run_full_workflow(
            client=client,
            topic=topic.strip(),
            material=material.strip(),
            level=level,
            goals=goals.strip(),
            study_days=study_days,
            mcq_count=mcq_count,
            flashcard_count=flashcard_count,
            model=model.strip() or "gpt-4.1-mini",
        )

        progress.progress(100)
        status.success("All five workflow stages completed.")

        st.session_state["workflow"] = workflow
        st.session_state["topic"] = topic.strip()
        st.session_state["level"] = level

    except Exception as exc:
        st.error(
            "The workflow stopped because of an error.\n\n"
            f"Details: {exc}"
        )


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

if "workflow" in st.session_state:
    workflow = st.session_state["workflow"]
    current_topic = st.session_state["topic"]
    current_level = st.session_state["level"]

    st.divider()
    st.subheader("2️⃣ AI Workflow Results")

    # Workflow stage visualization
    stages = [
        ("planning", "① Planning"),
        ("content_generation", "② Content Generation"),
        ("assessment", "③ Assessment"),
        ("review", "④ Review"),
        ("refinement", "⑤ Refinement"),
    ]

    stage_cols = st.columns(5)

    for column, (key, label) in zip(stage_cols, stages):
        with column:
            st.success(label)

    # Detailed stage context
    st.subheader("🔄 Context Passed Between Stages")

    for key, label in stages:
        with st.expander(label):
            st.json(workflow[key])

    # Assessment dashboard
    assessment = workflow["assessment"]
    review = workflow["review"]

    st.subheader("📊 AI Quality Dashboard")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Quality Score",
        f"{assessment.get('overall_score', 0)}/100",
    )

    c2.metric(
        "Automated QA",
        "PASS" if assessment.get("pass") else "REVIEW",
    )

    c3.metric(
        "Review Decision",
        str(review.get("decision", "N/A")).upper(),
    )

    if assessment.get("critical_issues"):
        st.warning("Critical issues identified:")
        for issue in assessment["critical_issues"]:
            st.write(f"- {issue}")

    validation_errors = workflow.get("validation_errors", [])

    if validation_errors:
        st.error("Final deterministic validation found problems:")
        for error in validation_errors:
            st.write(f"- {error}")
    else:
        st.success("Final deterministic validation passed.")

    # --------------------------------------------------------
    # Final study pack
    # --------------------------------------------------------

    final = workflow["refinement"]

    st.divider()
    st.subheader("3️⃣ Final Personalized Study Pack")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📘 Notes",
            "❓ Interactive MCQs",
            "🧠 Flashcards",
            "📝 Short Questions",
            "⬇️ Download",
        ]
    )

    with tab1:
        st.markdown("### Topic Overview")
        st.markdown(final.get("overview", ""))

        st.markdown("### Detailed Notes")
        st.markdown(final.get("detailed_notes", ""))

        st.markdown("### Quick Revision")
        for item in final.get("quick_revision", []):
            st.markdown(f"- {item}")

        st.markdown("### Key Terms")
        for item in final.get("key_terms", []):
            st.markdown(
                f"**{item.get('term', '')}:** "
                f"{item.get('definition', '')}"
            )

        st.markdown("### Examples")
        for item in final.get("examples", []):
            st.markdown(f"- {item}")

        st.markdown("### Study Plan")
        for item in final.get("study_plan", []):
            st.markdown(f"- {item}")

    with tab2:
        st.markdown(
            "Answer each question and click **Check Answer**."
        )

        for index, mcq in enumerate(final.get("mcqs", []), start=1):
            st.markdown(
                f"### Question {index}: {mcq.get('question', '')}"
            )

            options = mcq.get("options", {})

            selected = st.radio(
                "Select an option:",
                ["A", "B", "C", "D"],
                format_func=lambda x, opts=options:
                    f"{x}. {opts.get(x, '')}",
                key=f"answer_{index}",
            )

            if st.button(
                f"Check Answer {index}",
                key=f"check_{index}",
            ):
                correct = mcq.get("answer")

                if selected == correct:
                    st.success("✅ Correct!")
                else:
                    st.error(
                        f"❌ Incorrect. Correct answer: {correct}"
                    )

                st.info(mcq.get("explanation", ""))

            st.divider()

    with tab3:
        for index, card in enumerate(
            final.get("flashcards", []),
            start=1,
        ):
            with st.expander(f"Flashcard {index}"):
                st.markdown(
                    f"**Q:** {card.get('question', '')}"
                )
                st.markdown(
                    f"**A:** {card.get('answer', '')}"
                )

    with tab4:
        for index, item in enumerate(
            final.get("short_questions", []),
            start=1,
        ):
            with st.expander(f"Question {index}"):
                st.markdown(
                    f"**Question:** {item.get('question', '')}"
                )
                st.markdown(
                    f"**Model Answer:** {item.get('answer', '')}"
                )

    with tab5:
        markdown = build_markdown(
            current_topic,
            current_level,
            workflow,
        )

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            current_topic,
        ).strip("_") or "study_pack"

        st.download_button(
            "⬇️ Download Study Pack",
            data=markdown.encode("utf-8"),
            file_name=f"{safe_name}_study_pack.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download Complete Workflow JSON",
            data=json.dumps(
                workflow,
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8"),
            file_name=f"{safe_name}_workflow.json",
            mime="application/json",
            use_container_width=True,
        )

st.divider()

st.caption(
    "AI Study Pack Generator | "
    "Planning → Generation → Assessment → Review → Refinement"
)

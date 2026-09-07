import streamlit as st
from workflow import StudyPackPipeline
from prompts import PROMPTS

st.set_page_config(
    page_title="AI Study Pack Workflow Generator",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Study Pack Generator")
st.caption("Multi-Stage Agentic Workflow: Planning ➔ Generation ➔ Audit ➔ Refinement")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API key")
    
    st.markdown("---")
    st.markdown("### 🔄 Multi-Stage Pipeline Stages")
    st.markdown("1. 🎯 **Planner**: Generates Pydantic structured blueprint")
    st.markdown("2. 📝 **Generator**: Builds study modules, flashcards & quiz")
    st.markdown("3. 🔍 **Reviewer**: Audits quality & compliance")
    st.markdown("4. ✨ **Refiner**: Polishes final visual markdown layout")

# Main Input Section
col_left, col_right = st.columns(2)

with col_left:
    topic = st.text_input("Study Topic / Subject:", placeholder="e.g., Quantum Computing Basics")

with col_right:
    goal = st.text_input("Learning Goal / Target Outcome:", placeholder="e.g., Prepare for an introductory computer science exam")

st.markdown("---")

if st.button("🚀 Execute Workflow Pipeline", type="primary", use_container_width=True):
    if not api_key:
        st.error("⚠️ Please enter a valid Gemini API Key in the sidebar.")
    elif not topic or not goal:
        st.warning("⚠️ Please fill in both the topic and learning goal fields.")
    else:
        pipeline = StudyPackPipeline(api_key=api_key)
        status_box = st.status("Executing Multi-Stage Agentic Workflow...", expanded=True)
        
        try:
            def update_status(message: str):
                status_box.write(f"✓ {message}")

            plan, final_content, review = pipeline.run_pipeline(
                topic=topic,
                goal=goal,
                prompts=PROMPTS,
                status_callback=update_status
            )
            
            status_box.update(label="🎉 Multi-Stage Pipeline Completed Successfully!", state="complete", expanded=False)
            
            # Display Results in Structured Tabs
            tab_content, tab_blueprint, tab_audit = st.tabs([
                "📚 Generated Study Pack",
                "🎯 Stage 1: Blueprint",
                "🔍 Stage 3: Quality Audit"
            ])
            
            with tab_content:
                st.markdown(final_content)
                st.download_button(
                    label="📥 Download Study Pack (.md)",
                    data=final_content,
                    file_name=f"{topic.lower().replace(' ', '_')}_study_pack.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                
            with tab_blueprint:
                st.subheader("Structured Curriculum Blueprint")
                st.json(plan.model_dump())
                
            with tab_audit:
                st.subheader("Quality Audit Verdict")
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Audit Status", "PASSED" if review.is_valid else "NEEDS REVISION")
                m_col2.metric("Quality Score", f"{review.quality_score}/10")
                
                st.info(f"**Auditor Feedback:** {review.feedback}")

        except Exception as e:
            status_box.update(label="❌ Pipeline Failed", state="error", expanded=True)
            st.error(f"Execution Error: {str(e)}")

import streamlit as st
import json

st.set_page_config(page_title="AI Avatar Hiring Simulator", layout="wide")

st.title("🧠 AI-to-AI Hiring Simulator")
st.markdown("Drop in a **Resume** and **JD** below to simulate 100 rounds of AI interviews between Candidate and Hiring Manager avatars.")

# Text inputs
jd_text = st.text_area("📄 Job Description (JD)", height=300, placeholder="Paste the JD here...")
resume_text = st.text_area("👤 Candidate Resume", height=300, placeholder="Paste the Resume here...")

if st.button("Run 100-Round AI Interview") and jd_text and resume_text:

    progress_bar = st.progress(0, text="Starting interview...")
    status_placeholder = st.empty()
    def run_simulation_with_progress(jd_text, resume_text, num_rounds=5):
        import time
        from simulator import run_simulation as real_run_simulation
        transcript = []
        total_score = 0
        interviewer_decisions = []
        candidate_decisions = []
        errors = []
        for i in range(1, num_rounds + 1):
            try:
                status_placeholder.info(f"Round {i}: Generating question...")
                q_prompt = real_run_simulation.__globals__['generate_hiring_manager_prompt'](jd_text)
                question = real_run_simulation.__globals__['model'].generate_content(q_prompt).text.strip()

                status_placeholder.info(f"Round {i}: Candidate answering...")
                a_prompt = real_run_simulation.__globals__['generate_candidate_prompt'](resume_text, question)
                answer = real_run_simulation.__globals__['model'].generate_content(a_prompt).text.strip()

                status_placeholder.info(f"Round {i}: Evaluating answer...")
                s_prompt = real_run_simulation.__globals__['generate_eval_prompt'](jd_text, answer)
                score_raw = real_run_simulation.__globals__['model'].generate_content(s_prompt).text.strip()
                try:
                    score = int(score_raw.split()[0])
                    score = max(1, min(10, score))
                except:
                    score = 6

                total_score += score

                # Per-round decisions
                round_transcript = transcript + [{"round": i, "question": question, "answer": answer, "score": score}]
                status_placeholder.info(f"Round {i}: Interviewer deciding...")
                interviewer_prompt = real_run_simulation.__globals__['generate_interviewer_decision_prompt'](jd_text, round_transcript)
                interviewer_decision_raw = real_run_simulation.__globals__['model'].generate_content(interviewer_prompt).text.strip().upper()
                interviewer_decision = 'YES' in interviewer_decision_raw
                interviewer_decisions.append(interviewer_decision)

                status_placeholder.info(f"Round {i}: Candidate deciding...")
                candidate_prompt = real_run_simulation.__globals__['generate_candidate_decision_prompt'](resume_text, round_transcript)
                candidate_decision_raw = real_run_simulation.__globals__['model'].generate_content(candidate_prompt).text.strip().upper()
                candidate_decision = 'YES' in candidate_decision_raw
                candidate_decisions.append(candidate_decision)

                transcript.append({
                    "round": i,
                    "question": question,
                    "answer": answer,
                    "score": score,
                    "interviewer_decision": interviewer_decision,
                    "candidate_decision": candidate_decision
                })
                progress_bar.progress(i/num_rounds, text=f"Completed round {i} of {num_rounds}")
            except Exception as e:
                errors.append(f"Round {i} error: {str(e)}")
                status_placeholder.error(f"Error in round {i}: {str(e)}")
                time.sleep(1)
        avg_score = total_score / num_rounds if num_rounds else 0
        interviewer_final = sum(interviewer_decisions) / num_rounds >= 0.5
        candidate_final = sum(candidate_decisions) / num_rounds >= 0.5
        feedback = (
            "Strong overall! You clearly demonstrated skills aligned with the JD."
            if avg_score >= 7.5 else
            "Some answers lacked depth or relevance. Consider tailoring your responses better to the role."
        )
        return {
            "transcript": transcript,
            "average_score": avg_score,
            "interviewer_decision": interviewer_final,
            "candidate_decision": candidate_final,
            "final_match": interviewer_final and candidate_final,
            "final_feedback": feedback,
            "transcript_json": json.dumps(transcript, indent=2),
            "errors": errors
        }

    with st.spinner("🧠 Spinning up avatars and simulating interviews..."):
        results = run_simulation_with_progress(jd_text, resume_text)

    st.success("✅ Interview Simulation Complete!")

    # Show the full interview exchange with per-round decisions
    st.markdown("## 🗣️ Interview Exchange")
    for r in results['transcript']:
        st.markdown(f"""
        **Round {r['round']}**
        - **Q:** {r['question']}
        - **A:** {r['answer']}
        - **Score:** {r['score']}/10
        - **Interviewer Decision:** {'YES ✅' if r['interviewer_decision'] else 'NO ❌'}
        - **Candidate Decision:** {'YES ✅' if r['candidate_decision'] else 'NO ❌'}
        """)

    # Show both agent final decisions and match result
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Interviewer Decision", "YES ✅" if results['interviewer_decision'] else "NO ❌")
    with col2:
        st.metric("Final Candidate Decision", "YES ✅" if results['candidate_decision'] else "NO ❌")
    with col3:
        st.metric("Final Match", "MATCHED 🤝" if results['final_match'] else "NO MATCH 🚫")

    st.markdown(f"**Average Score:** `{results['average_score']:.2f}`")
    st.markdown("**Candidate Feedback:**")
    st.info(results["final_feedback"])

    st.download_button("� Download Transcript (JSON)", results["transcript_json"], file_name="transcript.json")

    st.download_button("📥 Download Transcript (JSON)", results["transcript_json"], file_name="transcript.json")

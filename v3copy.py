import json
import asyncio
import os
from dotenv import load_dotenv
import nest_asyncio
import streamlit as st

# Apply the patch to allow nested event loops for asyncio compatibility
nest_asyncio.apply()

# --- Load environment variables and configure the Gemini model ---
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load .env file if it exists
load_dotenv()

# Use Streamlit secrets for API key management
try:
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in Streamlit secrets or a .env file.")
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()


# --- Model and Utility Functions ---
class GeminiModel:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model = genai.GenerativeModel(
            model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

    async def generate_content_async(self, prompt: str, temperature=0.0):
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        response = await self.model.generate_content_async(prompt, generation_config=generation_config)
        return response

model = GeminiModel()

def _extract_json(s: str, expected_type='array'):
    """Extracts the first valid JSON array or object from a string."""
    start_char, end_char = ('[', ']') if expected_type == 'array' else ('{', '}')
    start_index = s.find(start_char)
    if start_index == -1: return None
    depth = 0
    for i in range(start_index, len(s)):
        if s[i] == start_char: depth += 1
        elif s[i] == end_char: depth -= 1
        if depth == 0:
            json_slice = s[start_index : i + 1]
            try: return json.loads(json_slice)
            except json.JSONDecodeError: return None
    return None

async def model_generate_json(prompt: str, retries=3, expected_type='array'):
    """Wrapper to call the model and ensure a valid JSON response."""
    last_exc = None
    for attempt in range(retries):
        try:
            response = await model.generate_content_async(prompt)
            if not response.parts:
                 raise ValueError(f"Response blocked. Reason: {response.prompt_feedback.block_reason.name}")
            raw_text = response.text
            parsed_json = _extract_json(raw_text, expected_type)
            if parsed_json: return parsed_json
            last_exc = ValueError(f"Failed to parse valid JSON from model response after {attempt + 1} attempts. Raw text: '{raw_text}'")
        except Exception as e:
            last_exc = e
            await asyncio.sleep(1)
    raise last_exc if last_exc else RuntimeError("Model generation failed.")

# --- Agent Definitions for Turn-by-Turn Conversation ---

async def recruiter_agent(jd: str, resume: str, num_questions: int, history: list):
    """JD Agent's turn: Asks a set of questions for the current round."""
    history_str = json.dumps(history, indent=2)
    prompt = f"""
    You are the JD Agent, an interviewer. Your goal is to assess a candidate based on a Job Description.
    You have the conversation history so far. Ask the next set of exactly {num_questions} insightful questions.
    DO NOT repeat questions that are already in the history. Focus on new areas or deeper dives.

    Return ONLY a single JSON array of strings, with no other text.

    Job Description:---{jd}---
    Candidate Resume:---{resume}---
    Conversation History:---{history_str}---

    Example format: ["What was your most challenging project and why?", "How do you handle tight deadlines?"]
    """
    questions = await model_generate_json(prompt, expected_type='array')
    if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
        return questions
    raise ValueError("JD Agent failed to return a valid list of questions.")

async def candidate_answers(jd: str, resume: str, current_questions: list, history: list):
    """Candidate Agent's turn: Answers the latest set of questions."""
    history_str = json.dumps(history, indent=2)
    questions_str = json.dumps(current_questions)
    prompt = f"""
    You are the Candidate Agent, a job candidate. Your goal is to answer questions based on your Resume and the Job Description.
    You have the conversation history and the new questions for this round. Answer ONLY the new questions.
    
    Return a JSON array of objects, where each object contains the "question" you were asked and your "answer".
    IMPORTANT: Return ONLY a single valid JSON array of objects, with no other text.

    Job Description:---{jd}---
    Your Resume:---{resume}---
    Conversation History:---{history_str}---
    New Questions to Answer Now:---{questions_str}---

    Example format: [{{"question": "...", "answer": "..."}}, {{"question": "...", "answer": "..."}}]
    """
    answers = await model_generate_json(prompt, expected_type='array')
    if isinstance(answers, list) and all('question' in item and 'answer' in item for item in answers):
        return answers
    raise ValueError("Resume Agent failed to return a valid list of answers.")

async def interviewer_decision_agent(jd: str, transcript_so_far: list):
    """Makes a yes/no decision for the current round from the interviewer's perspective."""
    transcript_str = json.dumps(transcript_so_far, indent=2)
    prompt = f"""
    You are an Interviewer. Based on the conversation so far, decide if you want to continue with this candidate.
    Return ONLY a single JSON object with your "decision" ("YES" or "NO") and a brief "reason".

    Job Description: --- {jd} ---
    Conversation Transcript: --- {transcript_str} ---
    Example: {{"decision": "YES", "reason": "The candidate is providing strong, relevant examples."}}
    """
    return await model_generate_json(prompt, expected_type='object')

async def candidate_decision_agent(jd: str, transcript_so_far: list):
    """Makes a yes/no decision for the current round from the candidate's perspective."""
    transcript_str = json.dumps(transcript_so_far, indent=2)
    prompt = f"""
    You are a job Candidate. Based on the conversation so far, decide if you are still interested in this role.
    Return ONLY a single JSON object with your "decision" ("YES" or "NO") and a brief "reason".

    Job Description: --- {jd} ---
    Conversation Transcript: --- {transcript_str} ---
    Example: {{"decision": "YES", "reason": "The role seems to align well with my career goals."}}
    """
    return await model_generate_json(prompt, expected_type='object')

async def evaluation_agent(jd: str, resume: str, transcript: list):
    """Evaluates the full interview transcript after the conversation is complete."""
    transcript_str = json.dumps(transcript, indent=2)
    prompt = f"""
    Act as a senior hiring manager. Evaluate the candidate's performance based on the JD, resume, and the full interview transcript.
    Provide a single score from 1 (poor fit) to 10 (excellent fit).
    Return ONLY a single JSON object with your "score" and a brief "justification".

    Job Description: --- {jd} ---
    Candidate Resume: --- {resume} ---
    Full Interview Transcript: --- {transcript_str} ---

    Example: {{"score": 8, "justification": "The candidate consistently provided relevant examples."}}
    """
    return await model_generate_json(prompt, expected_type='object')

# --- Orchestration ---

def run_async(coro):
    """Helper function to run an async coroutine in a Streamlit context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

async def run_full_interview(jd: str, resume: str, num_conv_rounds: int, num_questions_per_round: int, progress_callback):
    """Orchestrates the turn-by-turn interview conversation and decision-making."""
    full_transcript = []
    round_results = []
    total_steps = (num_conv_rounds * 4) + 1  # Ask, Answer, Interviewer Decide, Candidate Decide per round + Final Eval

    for i in range(num_conv_rounds):
        current_step = i * 4
        # Recruiter Agent's Turn
        progress_callback((current_step + 1) / total_steps, f"Round {i+1}: Recruiter is asking questions...")
        questions = await recruiter_agent(jd, resume, num_questions_per_round, full_transcript)

        # Candidate Agent's Turn
        progress_callback((current_step + 2) / total_steps, f"Round {i+1}: Candidate is answering...")
        new_answers = await candidate_answers(jd, resume, questions, full_transcript)
        full_transcript.extend(new_answers)
        
        # Interviewer Decision Turn
        progress_callback((current_step + 3) / total_steps, f"Round {i+1}: Interviewer is deciding...")
        interviewer_dec = await interviewer_decision_agent(jd, full_transcript)

        # Candidate Decision Turn
        progress_callback((current_step + 4) / total_steps, f"Round {i+1}: Candidate is deciding...")
        candidate_dec = await candidate_decision_agent(jd, full_transcript)

        round_results.append({
            "round": i + 1,
            "qa_pairs": new_answers,
            "interviewer_decision": interviewer_dec,
            "candidate_decision": candidate_dec
        })

    # Final Evaluation
    progress_callback(0.95, "Compiling final evaluation...")
    evaluation = await evaluation_agent(jd, resume, full_transcript)
    
    # Calculate final yes/no decisions
    interviewer_yes_count = sum(1 for r in round_results if r["interviewer_decision"].get("decision") == "YES")
    candidate_yes_count = sum(1 for r in round_results if r["candidate_decision"].get("decision") == "YES")
    
    final_interviewer_decision = "YES" if interviewer_yes_count / num_conv_rounds >= 0.5 else "NO"
    final_candidate_decision = "YES" if candidate_yes_count / num_conv_rounds >= 0.5 else "NO"
    final_match = "MATCH" if final_interviewer_decision == "YES" and final_candidate_decision == "YES" else "NO MATCH"
    
    progress_callback(1.0, "Complete!")
    return {
        "round_results": round_results, 
        "final_score": evaluation,
        "final_interviewer_decision": final_interviewer_decision,
        "final_candidate_decision": final_candidate_decision,
        "final_match": final_match
    }

# --- Streamlit UI ---

st.set_page_config(page_title="AI Hiring Simulator", layout="wide")
st.title("🤖 Live AI-to-AI Hiring Simulator")
st.markdown("A turn-by-turn interview between a **JD Agent** (Interviewer) and a **Resume Agent** (Candidate).")

# Input fields
col1, col2 = st.columns(2)
with col1:
    jd_text = st.text_area("📄 **Job Description (JD)**", height=300, placeholder="Paste the full job description here...")
with col2:
    resume_text = st.text_area("👤 **Candidate Resume**", height=300, placeholder="Paste the candidate's full resume text here...")

# UI controls for conversation structure
st.sidebar.header("Interview Structure")
num_conv_rounds = st.sidebar.slider("Number of Conversation Rounds", min_value=1, max_value=5, value=2, step=1)
num_questions_per_round = st.sidebar.slider("Number of Questions per Round", min_value=1, max_value=5, value=2, step=1)

if st.button("▶️ Run Live Interview Simulation", use_container_width=True):
    if not jd_text or not resume_text:
        st.warning("Please provide both a Job Description and a Resume.")
    else:
        progress_bar = st.progress(0, text="Starting simulation...")
        
        def update_progress(value, text):
            progress_bar.progress(value, text=text)

        results = run_async(run_full_interview(jd_text, resume_text, num_conv_rounds, num_questions_per_round, update_progress))

        if results:
            st.success("✅ Conversation Complete!")
            
            # --- Final Results Display ---
            st.subheader("Final Verdict")
            
            score_data = results.get('final_score', {})
            score = score_data.get('score', 'N/A')
            justification = score_data.get('justification', 'No justification provided.')

            final_interviewer = results.get('final_interviewer_decision', 'N/A')
            final_candidate = results.get('final_candidate_decision', 'N/A')
            final_match = results.get('final_match', 'N/A')

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric(label="**Overall Score**", value=f"{score}/10")
            with res_col2:
                st.metric(label="**Final Interviewer Decision**", value=final_interviewer)
            with res_col3:
                st.metric(label="**Final Candidate Decision**", value=final_candidate)
            with res_col4:
                st.metric(label="**Final Result**", value=final_match)
            
            st.info(f"**Justification:** {justification}")

            # --- Transcript Display ---
            with st.expander("📝 View Full Interview Transcript & Round-by-Round Decisions", expanded=True):
                for round_data in results.get('round_results', []):
                    st.markdown(f"--- \n### **Round {round_data['round']}**")
                    
                    # Q&A Pairs
                    for item in round_data['qa_pairs']:
                        st.markdown(f"**❓ Recruiter asks:** {item['question']}")
                        st.markdown(f"**💬 Candidate answers:** {item['answer']}")
                    
                    st.markdown("---") # Visual separator
                    
                    # Per-round decisions
                    dec_col1, dec_col2 = st.columns(2)
                    with dec_col1:
                        interviewer_dec = round_data['interviewer_decision']
                        decision = interviewer_dec.get('decision', 'N/A')
                        reason = interviewer_dec.get('reason', '')
                        st.markdown(f"**Interviewer Decision:** `{decision}`")
                        st.caption(f"Reason: {reason}")
                    
                    with dec_col2:
                        candidate_dec = round_data['candidate_decision']
                        decision = candidate_dec.get('decision', 'N/A')
                        reason = candidate_dec.get('reason', '')
                        st.markdown(f"**Candidate Decision:** `{decision}`")
                        st.caption(f"Reason: {reason}")


import json
import asyncio
import pandas as pd
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
# Try to get the key from st.secrets first, then fall back to environment variables
try:
    # This is the recommended way for Streamlit deployment
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in Streamlit secrets or a .env file.")
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()


# --- Model and Utility Functions ---

# This class wrapper provides a consistent interface to the Gemini model
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

# Initialize the model that the agents will use
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
            last_exc = ValueError("Failed to parse valid JSON from model response.")
        except Exception as e:
            last_exc = e
            await asyncio.sleep(1)
    raise last_exc if last_exc else RuntimeError("Model generation failed.")

# --- Agent Definitions ---

async def question_generator_agent(jd: str, resume: str, num_questions: int):
    prompt = f"""Based on the Job Description and Resume, act as a hiring manager. Generate a JSON array of exactly {num_questions} insightful interview questions.
    IMPORTANT: Return ONLY a single valid JSON array of strings.
    Job Description: --- {jd} ---
    Candidate Resume: --- {resume} ---
    Example: ["What was your most challenging project and why?", "How do you handle tight deadlines?"]"""
    questions = await model_generate_json(prompt, expected_type='array')
    if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
        return questions
    raise ValueError("Question generator agent failed to return a valid list of strings.")

async def answer_generator_agent(jd: str, resume: str, questions: list):
    prompt = f"""Act as the job candidate. Based on your resume, answer each question below, using the Job Description as context.
    Return a JSON array of objects, each containing the "question" and your "answer".
    IMPORTANT: Return ONLY a single valid JSON array of objects.
    Job Description: --- {jd} ---
    Candidate Resume: --- {resume} ---
    Interview Questions: {json.dumps(questions)}
    Example: [{{"question": "...", "answer": "..."}}]"""
    transcript = await model_generate_json(prompt, expected_type='array')
    if isinstance(transcript, list) and all('question' in item and 'answer' in item for item in transcript):
        return transcript
    raise ValueError("Answer generator agent failed to return a valid transcript.")

async def evaluation_agent(jd: str, resume: str, transcript: list):
    transcript_str = json.dumps(transcript, indent=2)
    prompt = f"""Act as a senior hiring manager. Evaluate the candidate's performance based on the JD, resume, and interview transcript.
    Provide a single score from 1 (poor fit) to 10 (excellent fit).
    Return ONLY a single JSON object with your "score" and a brief "justification".
    Job Description: --- {jd} ---
    Candidate Resume: --- {resume} ---
    Interview Transcript: --- {transcript_str} ---
    Example: {{"score": 8, "justification": "The candidate provided strong examples aligned with job requirements."}}"""
    evaluation = await model_generate_json(prompt, expected_type='object')
    if isinstance(evaluation, dict) and 'score' in evaluation and 'justification' in evaluation:
        return evaluation
    raise ValueError("Evaluation agent failed to return a valid evaluation object.")

# --- Orchestration ---

def run_async(coro):
    """Helper function to run an async coroutine in a Streamlit context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

async def run_full_interview(jd: str, resume: str, num_questions: int, progress_callback):
    """Orchestrates the interview, updating a progress callback for the UI."""
    try:
        progress_callback(0.25, "Generating questions...")
        questions = await question_generator_agent(jd, resume, num_questions)
        
        progress_callback(0.50, "Generating answers...")
        transcript = await answer_generator_agent(jd, resume, questions)
        
        progress_callback(0.75, "Evaluating interview...")
        evaluation = await evaluation_agent(jd, resume, transcript)
        
        progress_callback(1.0, "Complete!")
        return {"transcript": transcript, **evaluation}
    except Exception as e:
        st.error(f"An error occurred during the simulation: {e}")
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="AI Hiring Simulator", layout="wide")
st.title("🧠 AI-to-AI Hiring Simulator")
st.markdown("An interactive tool using three distinct AI agents to simulate and evaluate a job interview.")

# Input fields
col1, col2 = st.columns(2)
with col1:
    jd_text = st.text_area("📄 **Job Description (JD)**", height=300, placeholder="Paste the full job description here...")
with col2:
    resume_text = st.text_area("👤 **Candidate Resume**", height=300, placeholder="Paste the candidate's full resume text here...")

num_questions = st.slider("**Number of Questions to Generate**", min_value=2, max_value=10, value=5, step=1)

if st.button("▶️ Run Interview Simulation", use_container_width=True):
    if not jd_text or not resume_text:
        st.warning("Please provide both a Job Description and a Resume.")
    else:
        # Progress bar and status text
        progress_bar = st.progress(0, text="Starting simulation...")
        
        def update_progress(value, text):
            progress_bar.progress(value, text=text)

        # Run the simulation
        results = run_async(run_full_interview(jd_text, resume_text, num_questions, update_progress))

        if results:
            st.success("✅ Simulation Complete!")
            
            # Display final score and justification
            score = results.get('score', 'N/A')
            justification = results.get('justification', 'No justification provided.')
            
            st.subheader("Final Evaluation")
            st.metric(label="**Overall Score**", value=f"{score}/10")
            st.info(f"**Justification:** {justification}")
            
            # Display the full Q&A transcript in an expander
            with st.expander("📝 View Full Interview Transcript"):
                for item in results.get('transcript', []):
                    st.markdown(f"**❓ Question:** {item['question']}")
                    st.markdown(f"**💬 Answer:** {item['answer']}")
                    st.divider()
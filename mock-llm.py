import streamlit as st
import google.generativeai as genai
import os
import json
import asyncio
import pandas as pd
import time
import traceback
import re
from dotenv import load_dotenv
import nest_asyncio

# --- Page and Model Configuration ---
st.set_page_config(
    page_title="Correct MockLLM Implementation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply the nest_asyncio patch to allow nested event loops,
# which is crucial for running asyncio smoothly within Streamlit's architecture.
nest_asyncio.apply()

# Load environment variables from a .env file if it exists
load_dotenv()

# --- Gemini API Configuration ---
try:
    # Use Streamlit secrets for robust deployment, fallback to env variables
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in st.secrets or a .env file.")
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring the Gemini API: {e}")
    st.stop()

# --- Model and Utility Functions ---

# Use a stable, widely available model like "gemini-pro" to avoid 404 errors.
MODEL_NAME = "gemini-2.5-pro" 

def sanitize_text(text: str) -> str:
    """
    NEW: Aggressively cleans text to remove special characters, emojis,
    and excessive whitespace before sending it to the model.
    """
    if not isinstance(text, str):
        return ""
    # Remove non-ASCII characters (like emojis, special symbols)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Replace multiple newlines, tabs, and spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def robust_json_parser(json_string: str) -> dict:
    """
    A more robust JSON parser that extracts a JSON object from a string,
    even if it's embedded in markdown or has surrounding text.
    """
    try:
        # Find the start of the JSON object
        json_start_index = json_string.find('{')
        if json_start_index == -1:
            return {}

        # Find the end of the JSON object by matching braces
        open_braces = 0
        for i, char in enumerate(json_string[json_start_index:]):
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            
            if open_braces == 0:
                json_end_index = json_start_index + i + 1
                json_block = json_string[json_start_index:json_end_index]
                return json.loads(json_block)
        return {}
    except (json.JSONDecodeError, IndexError):
        return {}


async def generate_json_from_gemini(prompt: str, retries=3, delay=5):
    """
    Robustly call Gemini in text mode and use a safe parser to extract JSON.
    This is more resilient to special characters in the input.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    for attempt in range(retries):
        try:
            # Generate as standard text, not forced JSON
            response = await model.generate_content_async(prompt)
            
            # Use the robust parser to find and load the JSON
            parsed_json = robust_json_parser(response.text)
            
            if parsed_json: # Check if the parser returned a non-empty dictionary
                return parsed_json
            else:
                raise ValueError("Could not extract a valid JSON object from the model's response.")

        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Model call or JSON parsing failed (attempt {attempt+1}/{retries}). Retrying in {delay}s... Error: {e}")
                await asyncio.sleep(delay)
            else:
                st.error(f"Failed to get a valid JSON response from the model after {retries} attempts.")
                st.code(f"Final Prompt: {prompt}\nFinal Error: {e}\nRaw Model Response: {response.text if 'response' in locals() else 'No response'}")
                raise

# Initialize Reflection Memory in session state to persist between runs
if 'reflection_memory' not in st.session_state:
    st.session_state.reflection_memory = []

# --- AGENT IMPLEMENTATIONS (following the paper) ---

async def generate_interview_question_agent(jd: str, resume: str, history: list, memory_context: str):
    # Aggressively sanitize raw text inputs
    safe_jd = sanitize_text(jd)
    safe_resume = sanitize_text(resume)
    history_str = json.dumps(history)
    
    prompt = f"""
    You are an AI Interviewer Agent. Your behavior is "Interview Question Raising".
    Respond ONLY with a single JSON object inside a markdown code block.

    **CRITICAL REQUIREMENTS:**
    1.  **Coherence:** The question must logically follow the conversation history.
    2.  **Relevance:** The question must be relevant to both the Job Description and the candidate's Resume.
    3.  **Diversity:** The question must be different from all previous questions.

    {memory_context}

    **Job Description:**
    {safe_jd}

    **Candidate Resume:**
    {safe_resume}

    **Conversation History:**
    {history_str}

    Example Response:
    ```json
    {{
        "question": "Your resume mentions leading a project with Redis; can you elaborate on the specific caching strategy you implemented and its impact on system latency?"
    }}
    ```
    """
    return await generate_json_from_gemini(prompt)

async def generate_candidate_answer_agent(jd: str, resume: str, history: list, question: str):
    # Aggressively sanitize raw text inputs
    safe_jd = sanitize_text(jd)
    safe_resume = sanitize_text(resume)
    history_str = json.dumps(history)

    prompt = f"""
    You are an AI Candidate Agent. Your behavior is "Interview Response Generation".
    Respond ONLY with a single JSON object inside a markdown code block.

    **CRITICAL REQUIREMENTS:**
    1.  **Coherence:** Your answer must be consistent with previous answers.
    2.  **Relevance:** Your answer must demonstrate skills from your resume relevant to the job.
    3.  **Authenticity:** Ground your answer in experiences listed on your resume.

    **Job Description:**
    {safe_jd}

    **Your Resume:**
    {safe_resume}

    **Conversation History:**
    {history_str}

    **Question to Answer:**
    "{question}"

    Example Response:
    ```json
    {{
        "answer": "In that project, I implemented a write-through caching strategy using Redis. This reduced database read operations by 70% and cut the average API response time from 300ms to under 80ms."
    }}
    ```
    """
    return await generate_json_from_gemini(prompt)

async def perform_interviewer_evaluation_agent(jd: str, resume: str, transcript: list):
    # Aggressively sanitize raw text inputs
    safe_jd = sanitize_text(jd)
    safe_resume = sanitize_text(resume)
    transcript_str = json.dumps(transcript, indent=2)

    prompt = f"""
    You are an AI Interviewer Agent. Your behavior is "Interview Performance Evaluation".
    Respond ONLY with a single JSON object inside a markdown code block.

    **EVALUATION METHODOLOGY:**
    1.  **Basic Score (s_r):** Score 0-100 on how well the Resume matches the Job Description.
    2.  **Interview Score (s_q):** Score 0-100 on the quality of the candidate's answers in the transcript.
    3.  **Final Decision:** Make a final hiring decision ('HIRE' or 'NO_HIRE').
    
    **Job Description:** {safe_jd}
    **Candidate Resume:** {safe_resume}
    **Interview Transcript:** {transcript_str}

    Example Response:
    ```json
    {{
        "basicScore": 85,
        "interviewScore": 92,
        "reasoning": "Strong resume match and excellent, data-driven answers during the interview.",
        "decision": "HIRE"
    }}
    ```
    """
    return await generate_json_from_gemini(prompt)

async def perform_candidate_evaluation_agent(jd: str, transcript: list):
    # Aggressively sanitize raw text inputs
    safe_jd = sanitize_text(jd)
    transcript_str = json.dumps(transcript, indent=2)

    prompt = f"""
    You are an AI Candidate Agent. Your behavior is "Job Position Evaluation".
    Respond ONLY with a single JSON object inside a markdown code block.

    **EVALUATION METHODOLOGY:**
    1.  **Interest Score:** Score 0-100 on your interest in this position.
    2.  **Confidence Score:** Score 0-100 on your confidence in succeeding in this role.
    3.  **Final Decision:** Decide if you would accept an offer ('ACCEPT' or 'REJECT').

    **Job Description:** {safe_jd}
    **Interview Transcript:** {transcript_str}

    Example Response:
    ```json
    {{
        "interestScore": 95,
        "confidenceScore": 90,
        "reasoning": "The role seems like a perfect next step for my career and aligns with my technical interests.",
        "decision": "ACCEPT"
    }}
    ```
    """
    return await generate_json_from_gemini(prompt)

async def generate_reflection_memory_agent(jd: str, resume: str, transcript: list):
    # Aggressively sanitize raw text inputs
    safe_jd = sanitize_text(jd)
    safe_resume = sanitize_text(resume)
    transcript_str = json.dumps(transcript, indent=2)
    
    prompt = f"""
    You are an AI Strategy Agent. Your behavior is "Reflection Memory Generation".
    This interview was a successful match. Summarize the experience to help future interviews.
    Respond ONLY with a single JSON object inside a markdown code block.

    **TASK:**
    Generate a concise, one-sentence "questioning strategy" that an interviewer could use for future candidates with similar profiles.

    **Job Description:** {safe_jd}
    **Candidate Resume:** {safe_resume}
    **Successful Transcript:** {transcript_str}

    Example Response:
    ```json
    {{
        "strategy": "For candidates with strong DevOps experience, focus on probing the real-world impact and metrics of their CI/CD implementations."
    }}
    ```
    """
    return await generate_json_from_gemini(prompt)

# --- Orchestration Engine ---
def find_similar_experience_in_memory(resume: str):
    """Finds a similar successful interview in memory to inform strategy."""
    if not st.session_state.reflection_memory:
        return ""
    # Use sanitized text for similarity comparison
    resume_keywords = set(sanitize_text(resume).lower().split())
    best_match = None
    max_similarity = 0.2
    for memory in st.session_state.reflection_memory:
        memory_keywords = set(sanitize_text(memory['resume']).lower().split())
        if not resume_keywords or not memory_keywords:
            continue
        similarity = len(resume_keywords.intersection(memory_keywords)) / len(resume_keywords.union(memory_keywords))
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = memory

    if best_match:
        return f"\n**Strategy Modification from Reflection Memory:** Apply this learned strategy: '{best_match['strategy']}'"
    return ""

async def run_simulation_for_candidate(jd: str, candidate_data: dict, num_questions: int, progress_container):
    """Orchestrates the full, correct MockLLM flow for a single candidate."""
    name = candidate_data['name']
    resume = candidate_data['resume']
    transcript = []

    progress_bar = progress_container.progress(0, text=f"Preparing interview for {name}...")
    memory_context = find_similar_experience_in_memory(resume)

    for i in range(num_questions):
        progress_bar.progress((i + 1) / (num_questions + 1), text=f"Interviewing {name}: Question {i+1}/{num_questions}")
        question_json = await generate_interview_question_agent(jd, resume, transcript, memory_context)
        question = question_json.get('question', 'Could you tell me about your experience?')
        answer_json = await generate_candidate_answer_agent(jd, resume, transcript, question)
        answer = answer_json.get('answer', 'I have relevant experience in this area.')
        transcript.append({"question": question, "answer": answer})

    progress_bar.progress(1.0, text=f"Performing two-sided evaluation for {name}...")
    interviewer_eval_task = perform_interviewer_evaluation_agent(jd, resume, transcript)
    candidate_eval_task = perform_candidate_evaluation_agent(jd, transcript)
    interviewer_eval, candidate_eval = await asyncio.gather(interviewer_eval_task, candidate_eval_task)

    is_handshake_success = (interviewer_eval.get('decision') == 'HIRE' and 
                            candidate_eval.get('decision') == 'ACCEPT')
    reflection = None
    if is_handshake_success:
        reflection_json = await generate_reflection_memory_agent(jd, resume, transcript)
        reflection = reflection_json.get('strategy')
        st.session_state.reflection_memory.append({
            "candidate_name": name, "resume": resume, 
            "transcript": transcript, "strategy": reflection
        })
        st.toast(f"✅ New strategy learned from {name} and added to Memory!")

    basic_score = interviewer_eval.get('basicScore') or 0
    interview_score = interviewer_eval.get('interviewScore') or 0
    final_score = (basic_score * 0.4) + (interview_score * 0.6)

    return {
        "name": name, "score": round(final_score),
        "handshake": "🤝 SUCCESSFUL MATCH" if is_handshake_success else "❌ NO MATCH",
        "interviewer_eval": interviewer_eval, "candidate_eval": candidate_eval,
        "transcript": transcript, "reflection_generated": reflection
    }

async def main_orchestrator(candidates, jd, num_q, overall_progress_bar, results_container):
    BATCH_SIZE = 8
    all_results = []
    processed_count = 0
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i:i + BATCH_SIZE]
        batch_number = (i // BATCH_SIZE) + 1
        total_batches = -(-len(candidates) // BATCH_SIZE)
        results_container.info(f"🚀 Processing Batch {batch_number}/{total_batches} ({len(batch)} candidates concurrently)...")
        progress_containers = {c['name']: results_container.empty() for c in batch}
        tasks = [run_simulation_for_candidate(jd, c, num_q, progress_containers[c['name']]) for c in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for candidate, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                results_container.error(f"Failed to process {candidate['name']}. Error: {result}")
            elif result:
                all_results.append(result)

        processed_count += len(batch)
        overall_progress_bar.progress(processed_count / len(candidates))
    return all_results

# --- STREAMLIT UI ---
st.title("✅ Correct MockLLM Implementation")
st.markdown("A robust demonstration of the MockLLM framework using gemini-pro, designed to handle complex, uncleaned resume text.")

st.sidebar.header("⚙️ Configuration")
num_questions = st.sidebar.slider("Number of Interview Questions", 1, 5, 2)
st.sidebar.markdown("---")
if st.sidebar.button("Clear Reflection Memory"):
    st.session_state.reflection_memory = []
    st.toast("Reflection Memory has been cleared.")

with st.expander("🧠 View Reflection Memory", expanded=False):
    if not st.session_state.reflection_memory:
        st.info("Reflection Memory is empty.")
    else:
        st.success(f"{len(st.session_state.reflection_memory)} strategies learned.")
        for i, mem in enumerate(st.session_state.reflection_memory):
            with st.container(border=True):
                st.markdown(f"Strategy #{i+1} (from {mem['candidate_name']})")
                st.info(f"Learned Strategy: {mem['strategy']}")

jd_col, uploader_col = st.columns([2, 1])
with jd_col:
    jd_text = st.text_area("📄 Job Description", height=300, value="Position: Senior Python Developer\nRequirements: 5+ years experience with Python, Django, and AWS. Must have experience leading a team.")
with uploader_col:
    uploaded_file = st.file_uploader("👤 Upload Candidates CSV", type="csv")
    st.markdown("CSV must have 'name' and 'resume' columns.")

if st.button("🚀 Run MockLLM Simulation", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("Please upload a CSV file with candidate data.")
    elif not jd_text:
        st.warning("Please provide a Job Description.")
    else:
        try:
            # Explicitly specify 'utf-8' encoding to handle a wider range of characters
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            if 'name' not in df.columns or 'resume' not in df.columns:
                st.error("CSV file must contain 'name' and 'resume' columns.")
            else:
                # Ensure data is string type and handle potential empty rows
                df = df.dropna(subset=['name', 'resume'])
                df['name'] = df['name'].astype(str)
                df['resume'] = df['resume'].astype(str)
                candidates = df.to_dict('records')

                st.info(f"Starting simulation for {len(candidates)} candidate(s)...")
                overall_progress = st.progress(0)
                results_container = st.container()
                start_time = time.time()
                all_results = asyncio.run(main_orchestrator(candidates, jd_text, num_questions, overall_progress, results_container))
                end_time = time.time()
                st.success(f"Simulation for all candidates completed in {end_time - start_time:.2f} seconds.")
                if all_results:
                    all_results.sort(key=lambda x: x['score'], reverse=True)
                    
                    # Prepare CSV data for download
                    csv_data = []
                    for rank, res in enumerate(all_results, 1):
                        csv_data.append({
                            "Rank": rank,
                            "Name": res['name'],
                            "Final Score": res['score'],
                            "Handshake Result": res['handshake'],
                            "Basic Score": res['interviewer_eval'].get('basicScore', 0),
                            "Interview Score": res['interviewer_eval'].get('interviewScore', 0),
                            "Interviewer Decision": res['interviewer_eval'].get('decision', 'N/A'),
                            "Candidate Decision": res['candidate_eval'].get('decision', 'N/A'),
                            "Interviewer Reasoning": res['interviewer_eval'].get('reasoning', 'N/A'),
                            "Candidate Reasoning": res['candidate_eval'].get('reasoning', 'N/A'),
                            "Strategy Learned": res['reflection_generated'] if res['reflection_generated'] else 'N/A'
                        })
                    
                    results_df = pd.DataFrame(csv_data)
                    csv_string = results_df.to_csv(index=False)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_string,
                        file_name=f"mockllm_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    st.markdown("--- \n## 🏆 Final Candidate Rankings")
                    for rank, res in enumerate(all_results, 1):
                        st.markdown(f"### **Rank #{rank}: {res['name']}**")
                        with st.container(border=True):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Final Score", f"{res['score']}/100")
                            c2.metric("Handshake Result", res['handshake'])
                            if res['reflection_generated']: 
                                c3.info("🧠 Strategy Learned!")
                            with st.expander(f"View Detailed Analysis for {res['name']}"):
                                st.subheader("📝 Mock Interview Transcript")
                                for turn in res['transcript']:
                                    st.markdown(f"**Interviewer:** {turn['question']}\n\n**{res['name']}:** {turn['answer']}\n\n---")
                                st.subheader("🤝 Two-Sided Evaluation")
                                eval1, eval2 = st.columns(2)
                                with eval1:
                                    st.markdown("**Interviewer Evaluation**")
                                    st.json(res['interviewer_eval'])
                                with eval2:
                                    st.markdown("**Candidate Evaluation**")
                                    st.json(res['candidate_eval'])
                                if res['reflection_generated']:
                                    st.subheader("🧠 Reflection Memory Generated")
                                    st.success(f"**Learned Strategy:** {res['reflection_generated']}")
        except Exception as e:
            st.error(f"An unexpected error occurred during the simulation: {e}")
            st.code(traceback.format_exc())
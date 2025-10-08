import json
import asyncio
import os
from dotenv import load_dotenv
import nest_asyncio
import streamlit as st
import time
import traceback

# Apply the patch to allow nested event loops for asyncio compatibility with Streamlit
nest_asyncio.apply()

# --- Load environment variables and configure the Gemini model ---
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load .env file if it exists
load_dotenv()

# Use Streamlit secrets for API key management for robust deployment
try:
    # Safely get the API key from Streamlit secrets first, then fall back to environment variables
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in Streamlit secrets or a .env file.")
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# --- Model and Utility Functions ---
class GeminiModel:
    """A wrapper for the Gemini API model for easy configuration."""
    def __init__(self, model_name="gemini-2.5-flash"): # Using a modern, fast model
        self.model = genai.GenerativeModel(
            model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

    async def generate_content_async(self, prompt: str, temperature=0.2): # Slightly increased for creativity
        """Generates content asynchronously."""
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        response = await self.model.generate_content_async(prompt, generation_config=generation_config)
        return response

# Instantiate the model for use by all agents
model = GeminiModel()

def _extract_json(s: str, expected_type='object'):
    """Robustly extracts the first valid JSON object or array from a string."""
    start_char, end_char = ('[', ']') if expected_type == 'array' else ('{', '}')
    start_index = s.find(start_char)
    if start_index == -1: return None
    
    depth = 0
    for i in range(start_index, len(s)):
        if s[i] == start_char: depth += 1
        elif s[i] == end_char: depth -= 1
        if depth == 0:
            json_slice = s[start_index : i + 1]
            try:
                return json.loads(json_slice)
            except json.JSONDecodeError:
                # Continue searching if the slice is invalid
                pass
    return None

async def model_generate_json(prompt: str, retries=3, expected_type='object'):
    """
    Wrapper to call the model, automatically retry, and ensure a valid JSON response.
    This function enhances reliability by handling API errors and malformed JSON.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            response = await model.generate_content_async(prompt)
            if not response.parts:
                reason = response.prompt_feedback.block_reason.name if response.prompt_feedback else "UNKNOWN"
                raise ValueError(f"Response blocked by safety filters. Reason: {reason}")
            
            raw_text = response.text
            parsed_json = _extract_json(raw_text, expected_type)
            
            if parsed_json:
                return parsed_json
            else:
                 last_exc = ValueError(f"Failed to parse valid JSON from model output. Raw text: '{raw_text}'")

        except Exception as e:
            last_exc = e
            # Exponential backoff for retries
            await asyncio.sleep(1.5 ** attempt)
            
    raise last_exc if last_exc else RuntimeError("Model generation failed after multiple retries.")

# --- "Sub-Agents" for Parallel Execution ---

async def pre_screening_agent(jd: str, resume: str):
    """
    **NEW AGENT**
    Acts as a pre-screener to check for fundamental mismatches like years of experience.
    This runs before the interview to avoid wasting time on clearly unqualified candidates.
    """
    prompt = f"""
    Act as a strict HR pre-screener. Your only job is to check for a match on the explicit 'years of experience' requirement.
    - Read the required years of experience from the Job Description.
    - **If the Job Description does NOT mention a specific number of years of experience, you MUST assume it is not a hard requirement. In this case, give a score of 100 and state that no specific experience duration was mentioned.**
    - If a number is mentioned, calculate the candidate's total years of experience from the Resume.
    - If the candidate meets or exceeds the required years, give a score of 100.
    - If the candidate does NOT meet the required years, give a score below 40 and explicitly state the mismatch in the justification.
    - Return ONLY a single JSON object with your "score" and a brief "justification".

    Job Description:
    ---
    {jd}
    ---
    Candidate Resume:
    ---
    {resume}
    ---
    Example (Mismatch): {{"score": 30, "justification": "JD requires 10+ years of experience, but the resume shows approximately 4 years. This is a hard requirement mismatch."}}
    Example (Match): {{"score": 100, "justification": "Candidate meets the minimum years of experience requirement."}}
    Example (Not Mentioned): {{"score": 100, "justification": "The JD does not specify a required number of years of experience, so this is not a blocking factor."}}
    """
    return await model_generate_json(prompt, expected_type='object')


async def recruiter_panel_agent(jd: str, resume: str, history: list):
    """Generates ONE insightful interview question, avoiding repetition."""
    history_str = json.dumps(history, indent=2)
    prompt = f"""
    You are an expert technical recruiter on an interview panel. Your task is to ask ONE unique and insightful question to assess a candidate's skills against the job description.
    - DO NOT repeat any questions from the conversation history.
    - Ask a mix of technical questions to validate resume claims and behavioral (soft skill) questions to assess teamwork, leadership, and problem-solving.
    - The question should be probing and relevant to the role.
    - Return ONLY a single JSON object with a "question" key.

    Job Description:
    ---
    {jd}
    ---
    Candidate Resume:
    ---
    {resume}
    ---
    Conversation History (questions already asked):
    ---
    {history_str}
    ---
    Example format: {{"question": "Can you describe a time you had a disagreement with a team member and how you resolved it?"}}
    """
    result = await model_generate_json(prompt, expected_type='object')
    if isinstance(result, dict) and 'question' in result:
        return result['question']
    raise ValueError("Recruiter agent failed to return a valid question object.")

async def candidate_panel_agent(jd: str, resume: str, history: list, question: str):
    """Generates an answer to ONE question, acting as the candidate."""
    history_str = json.dumps(history, indent=2)
    prompt = f"""
    You are the Candidate Agent. Your persona is a top-tier professional, perfectly matching the job description.
    - You must provide an exemplary, high-scoring answer to the single question provided.
    - Use the STAR (Situation, Task, Action, Result) method, especially for behavioral questions.
    - Quantify your results with metrics whenever possible.
    - **Crucially, ground your answers in the provided resume.** For soft skill questions (e.g., about leadership or teamwork), infer context from your project roles and accomplishments. For instance, if the resume says "Led the development of a real-time sentiment analysis platform," use that specific project to describe your leadership style. Do not invent experiences not supported by the resume.
    - Return a JSON object containing the original "question" and your "answer".
    - IMPORTANT: Return ONLY a single valid JSON object.

    Job Description:
    ---
    {jd}
    ---
    Your Resume:
    ---
    {resume}
    ---
    Conversation History:
    ---
    {history_str}
    ---
    Question to Answer Now:
    ---
    {question}
    ---
    Example format: {{"question": "...", "answer": "..."}}
    """
    result = await model_generate_json(prompt, expected_type='object')
    if isinstance(result, dict) and 'question' in result and 'answer' in result:
        return result
    # Fallback to ensure the simulation doesn't crash
    return {"question": question, "answer": f"Could not generate a structured answer. Raw model output: {result}"}

# --- Round-End Agents (Evaluation and Decision-Making) ---

async def interviewer_decision_agent(jd: str, transcript_so_far: list):
    """Decides whether to continue the interview from the interviewer's perspective."""
    transcript_str = json.dumps(transcript_so_far, indent=2)
    prompt = f"""
    You are a Hiring Manager. Based on the conversation so far, should you continue with this candidate?
    - Return ONLY a single JSON object with your "decision" ("YES" or "NO") and a brief "reason".
    
    Job Description:
    ---
    {jd}
    ---
    Conversation Transcript:
    ---
    {transcript_str}
    ---
    Example: {{"decision": "YES", "reason": "The candidate is providing strong, relevant examples that directly map to the job requirements."}}
    """
    return await model_generate_json(prompt, expected_type='object')

async def candidate_decision_agent(jd: str, transcript_so_far: list):
    """Decides whether to continue from the candidate's perspective."""
    transcript_str = json.dumps(transcript_so_far, indent=2)
    prompt = f"""
    You are the job Candidate. Based on the conversation so far, are you still interested in this role?
    - Return ONLY a single JSON object with your "decision" ("YES" or "NO") and a brief "reason".
    
    Job Description:
    ---
    {jd}
    ---
    Conversation Transcript:
    ---
    {transcript_str}
    ---
    Example: {{"decision": "YES", "reason": "The questions are insightful and the role seems to align perfectly with my career goals and expertise."}}
    """
    return await model_generate_json(prompt, expected_type='object')

async def round_evaluation_agent(jd: str, resume: str, transcript: list):
    """Evaluates the interview transcript at the end of a round."""
    transcript_str = json.dumps(transcript, indent=2)
    prompt = f"""
    Act as an unbiased Hiring Manager. Evaluate the candidate's performance based on the interview transcript *so far*.
    - Provide a score from 1 (poor fit) to 100 (excellent fit).
    - Your justification should be concise and directly reference the transcript. Do not consider years of experience yet, focus on the quality of the answers.
    - Return ONLY a single JSON object with your "score" and "justification".

    Job Description:
    ---
    {jd}
    ---
    Candidate Resume:
    ---
    {resume}
    ---
    Interview Transcript So Far:
    ---
    {transcript_str}
    ---
    Example: {{"score": 95, "justification": "Candidate provided outstanding, metric-driven examples that directly align with the core responsibilities outlined in the JD."}}
    """
    return await model_generate_json(prompt, expected_type='object')


async def final_evaluation_agent(jd: str, resume: str, transcript: list):
    """Provides a final, comprehensive evaluation of the entire interview."""
    transcript_str = json.dumps(transcript, indent=2)
    prompt = f"""
    Act as the final decision-maker, a senior hiring manager. Your evaluation is the most important one.
    - Evaluate the candidate's performance based on the full interview transcript.
    - **Crucially, you MUST be strict about hard requirements mentioned in the Job Description. If a requirement like '10+ years of experience' is mentioned and the candidate does not meet it, the score MUST be low (e.g., under 50), regardless of how well they answered other questions.**
    - **If the JD does NOT specify a number for years of experience, evaluate the candidate based on the quality of their experience and answers alone, without penalizing them for duration.**
    - Provide a final score from 1 (poor fit) to 100 (excellent fit).
    - The justification should summarize the candidate's overall performance and explicitly mention any hard requirement mismatches if they exist.
    - Return ONLY a single JSON object with your "score" and a final "justification".

    Job Description:
    ---
    {jd}
    ---
    Candidate Resume:
    ---
    {resume}
    ---
    Full Interview Transcript:
    ---
    {transcript_str}
    ---
    Example (Mismatch): {{"score": 30, "justification": "While the candidate has strong skills, the JD strictly requires 10+ years of experience and the candidate only has 4. This is a critical mismatch."}}
    Example (Match): {{"score": 95, "justification": "The candidate's experience and strong answers make them an excellent fit for this role."}}
    """
    return await model_generate_json(prompt, expected_type='object')


# --- Orchestration Engine ---

def run_async(coro):
    """Helper function to run an async coroutine in a synchronous Streamlit environment."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

async def run_full_interview(jd: str, resume: str, num_conv_rounds: int, num_questions_per_round: int, progress_callback):
    """Orchestrates the entire interview, starting with a pre-screening check."""
    
    # **NEW**: Pre-screening step
    progress_callback(0.05, "Pre-screening for basic qualifications like years of experience...")
    pre_screen_result = await pre_screening_agent(jd, resume)
    pre_screen_score = pre_screen_result.get('score', 0)

    # If pre-screening fails significantly, end the process early.
    if pre_screen_score < 50:
        progress_callback(1.0, "Pre-screening failed. Simulation complete.")
        return {
            "round_results": [],
            "final_score": pre_screen_result, # Use the pre-screen result as the final score
            "final_interviewer_decision": "NO",
            "final_candidate_decision": "N/A", # Candidate doesn't get a say
            "final_match": "NO MATCH (Pre-screen failed)"
        }


    full_transcript = []
    round_results = []
    total_steps = (num_conv_rounds * 3) + 2 # Pre-screen, 3 steps per round, 1 final eval

    for i in range(num_conv_rounds):
        current_step_base = i * 3 + 1 # Offset by 1 for pre-screening step
        
        # --- ROUND START ---
        
        # 1. Ask Questions in Parallel
        progress_callback((current_step_base + 1) / total_steps, f"Round {i+1}: Panel is asking {num_questions_per_round} questions simultaneously...")
        question_tasks = [recruiter_panel_agent(jd, resume, full_transcript) for _ in range(num_questions_per_round)]
        questions_list = await asyncio.gather(*question_tasks)
        unique_questions = list(dict.fromkeys(questions_list)) # Remove duplicate questions

        # 2. Answer Questions in Parallel
        progress_callback((current_step_base + 2) / total_steps, f"Round {i+1}: Candidate is answering {len(unique_questions)} questions simultaneously...")
        answer_tasks = [candidate_panel_agent(jd, resume, full_transcript, q) for q in unique_questions]
        new_answers = await asyncio.gather(*answer_tasks)
        full_transcript.extend(new_answers)
        
        # 3. Make Decisions & Evaluate Round in Parallel
        progress_callback((current_step_base + 3) / total_steps, f"Round {i+1}: Evaluating round and making decisions...")
        end_of_round_tasks = [
            interviewer_decision_agent(jd, full_transcript),
            candidate_decision_agent(jd, full_transcript),
            round_evaluation_agent(jd, resume, full_transcript) # Evaluate this round
        ]
        interviewer_dec, candidate_dec, round_eval = await asyncio.gather(*end_of_round_tasks)

        round_results.append({
            "round": i + 1,
            "qa_pairs": new_answers,
            "interviewer_decision": interviewer_dec,
            "candidate_decision": candidate_dec,
            "round_evaluation": round_eval,
        })
        
        # --- ROUND END ---

    # 4. Final Evaluation (Runs once after all rounds are complete)
    progress_callback(0.95, "Compiling final evaluation...")
    final_eval = await final_evaluation_agent(jd, resume, full_transcript)
    final_score = final_eval.get("score", 0)
    
    # **REVISED**: Final decision logic is now driven by the final score
    candidate_yes_count = sum(1 for r in round_results if r["candidate_decision"].get("decision") == "YES")
    
    # The final evaluation is the deciding factor. If score is low, it's a "NO".
    if final_score < 60: # Setting a threshold for a match
        final_interviewer_decision = "NO"
    else:
        # If the score is good, then we can consider it a pass from the interviewer.
        final_interviewer_decision = "YES"

    final_candidate_decision = "YES" if num_conv_rounds > 0 and candidate_yes_count / num_conv_rounds >= 0.5 else "NO"
    
    # The final match depends on the STRICT interviewer decision and candidate's interest
    final_match = "MATCH" if final_interviewer_decision == "YES" and final_candidate_decision == "YES" else "NO MATCH"
    
    progress_callback(1.0, "Simulation Complete!")
    return {
        "round_results": round_results, 
        "final_score": final_eval,
        "final_interviewer_decision": final_interviewer_decision,
        "final_candidate_decision": final_candidate_decision,
        "final_match": final_match
    }

# --- Streamlit UI ---

# Default text for a perfect score scenario
DEFAULT_JD = """
**Position: Senior AI Engineer**

**Location:** Bengaluru, Karnataka, India

**Company:** InnovateAI Corp

**Description:**
We are seeking a highly skilled Senior AI Engineer to join our dynamic team. The ideal candidate will have a strong background in developing and deploying machine learning models, with a focus on large language models (LLMs) and cloud-native applications. You will be responsible for the end-to-end lifecycle of our AI products, from data preprocessing and model training to deployment, monitoring, and optimization in a production environment.

**Responsibilities:**
- Design, build, and maintain scalable machine learning pipelines using Python, Docker, and Kubernetes.
- Fine-tune and deploy large language models (LLMs) for various NLP tasks.
- Develop and manage high-performance APIs using FastAPI for model serving.
- Implement and monitor CI/CD pipelines for ML models using GitHub Actions.
- Collaborate with cross-functional teams to define problem statements and deliver AI-driven solutions.
- Optimize model performance for latency and throughput on cloud platforms (GCP/AWS).

**Qualifications:**
- 5+ years of experience in a software engineering or machine learning role.
- Proven experience with Python and frameworks like TensorFlow or PyTorch.
- Strong hands-on experience with FastAPI for building production-ready APIs.
- Expertise in containerization (Docker) and orchestration (Kubernetes).
- Demonstrable experience with deploying and managing ML models in a cloud environment (GCP preferred).
- Experience with CI/CD tools, specifically GitHub Actions.
"""

DEFAULT_RESUME = """
**Name:** Priya Sharma
**Contact:** priya.sharma@email.com | LinkedIn: /in/priyasharma-ai

**Summary:**
Results-oriented Senior AI Engineer with over 6 years of experience in designing, developing, and deploying end-to-end machine learning solutions. Expert in Python, Large Language Models, and cloud-native infrastructure, with a proven track record of delivering high-impact AI products that drive business value.

**Experience:**

**Senior Machine Learning Engineer | TechSolutions Inc. | Bengaluru, IN**
(2020 - Present)
- Led the development of a real-time sentiment analysis platform using a fine-tuned BERT model, deployed as a microservice on Google Kubernetes Engine (GKE).
- Engineered a high-throughput API using FastAPI, which served model predictions with a p99 latency of under 150ms, handling over 1 million requests per day.
- Architected and implemented a full CI/CD pipeline with GitHub Actions for automated model training, validation, and deployment, reducing the release cycle time by 60%.
- Containerized all ML applications using Docker, ensuring consistent and reproducible environments from development to production.

**Machine Learning Engineer | DataDriven Co. | Bengaluru, IN**
(2018 - 2020)
- Developed a recommendation engine using collaborative filtering techniques in Python, increasing user engagement by 25%.
- Worked on data preprocessing and feature engineering pipelines for large-scale datasets on GCP.
- Gained foundational experience with Docker and early-stage ML model deployment.

**Skills:**
- **Programming:** Python (Expert), SQL
- **ML/DL Frameworks:** PyTorch, TensorFlow, Scikit-learn, Transformers
- **API Development:** FastAPI, Flask
- **DevOps/MLOps:** Docker, Kubernetes, GitHub Actions, Terraform
- **Cloud:** Google Cloud Platform (GCP), AWS
"""

st.set_page_config(page_title="AI Hiring Simulator", layout="wide", initial_sidebar_state="expanded")
st.title("⚡️ High-Speed AI Hiring Simulator")
st.markdown("This app simulates a multi-round interview where all questions, answers, and evaluations in a round are processed simultaneously using parallel API calls to Gemini.")

col1, col2 = st.columns(2)
with col1:
    jd_text = st.text_area("📄 **Job Description (JD)**", height=300, value=DEFAULT_JD)
with col2:
    resume_text = st.text_area("👤 **Candidate Resume**", height=300, value=DEFAULT_RESUME)

st.sidebar.header("Interview Configuration")
num_conv_rounds = st.sidebar.slider("Number of Interview Rounds", min_value=1, max_value=5, value=2, step=1)
num_questions_per_round = st.sidebar.slider("Simultaneous Questions per Round", min_value=1, max_value=5, value=2, step=1)

if st.button("▶️ Run High-Speed Simulation", use_container_width=True, type="primary"):
    if not jd_text or not resume_text:
        st.warning("Please provide both a Job Description and a Resume.")
    else:
        start_time = time.time()
        progress_bar = st.progress(0, text="Initializing simulation...")
        
        def update_progress(value, text):
            progress_bar.progress(value, text=text)

        try:
            results = run_async(run_full_interview(jd_text, resume_text, num_conv_rounds, num_questions_per_round, update_progress))
            
            end_time = time.time()
            st.info(f"Simulation completed in {end_time - start_time:.2f} seconds.")

            if results:
                st.success("✅ Interview Simulation Complete!")
                st.subheader("Final Verdict")
                
                score_data = results.get('final_score', {})
                score = score_data.get('score', 'N/A')
                justification = score_data.get('justification', 'No justification provided.')
                final_interviewer = results.get('final_interviewer_decision', 'N/A')
                final_candidate = results.get('final_candidate_decision', 'N/A')
                final_match = results.get('final_match', 'N/A')

                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1: st.metric(label="**Overall Score**", value=f"{score}/100")
                with res_col2: st.metric(label="**Hiring Decision**", value=final_interviewer)
                with res_col3: st.metric(label="**Candidate Interest**", value=final_candidate)
                with res_col4: st.metric(label="**Final Match**", value=final_match)
                
                st.info(f"**Final Justification:** {justification}")

                with st.expander("📝 View Full Interview Transcript & Round-by-Round Analysis", expanded=True):
                    # Handle case where pre-screening fails and there are no rounds
                    if not results.get('round_results'):
                        st.warning("Interview did not proceed past the pre-screening stage.")
                    
                    for round_data in results.get('round_results', []):
                        st.markdown(f"--- \n### **Round {round_data['round']}**")
                        for item in round_data['qa_pairs']:
                            st.markdown(f"**❓ Interviewer asks:** {item['question']}")
                            st.markdown(f"**💬 Candidate answers:** {item['answer']}")
                        
                        st.markdown("---")
                        
                        # Display Round Decisions
                        dec_col1, dec_col2 = st.columns(2)
                        with dec_col1:
                            interviewer_dec = round_data['interviewer_decision']
                            st.markdown(f"**Interviewer Decision:** `{interviewer_dec.get('decision', 'N/A')}`")
                            st.caption(f"Reason: {interviewer_dec.get('reason', '')}")
                        
                        with dec_col2:
                            candidate_dec = round_data['candidate_decision']
                            st.markdown(f"**Candidate Decision:** `{candidate_dec.get('decision', 'N/A')}`")
                            st.caption(f"Reason: {candidate_dec.get('reason', '')}")
                        
                        # Display Round Evaluation
                        st.markdown("---")
                        round_eval = round_data.get('round_evaluation', {})
                        round_score = round_eval.get('score', 'N/A')
                        round_justification = round_eval.get('justification', 'No justification provided.')
                        
                        st.metric(label=f"**Score After Round {round_data['round']}**", value=f"{round_score} / 100")
                        st.caption(f"Justification: {round_justification}")

        except Exception as e:
            st.error(f"An error occurred during the simulation: {e}")
            st.code(traceback.format_exc())


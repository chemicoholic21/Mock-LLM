def generate_interviewer_decision_prompt(jd_text, transcript):
    transcript_str = "\n".join([
        f"Q: {r['question']}\nA: {r['answer']} (Score: {r['score']}/10)" for r in transcript
    ])
    return f'''
You are a highly experienced hiring manager. Based on the job description and the following interview transcript, decide if you would hire this candidate. Respond with only YES or NO.

Job Description:\n"""
{jd_text}
"""

Interview Transcript:\n"""
{transcript_str}
"""

Would you hire this candidate? (YES or NO):
'''.strip()

def generate_candidate_decision_prompt(resume_text, transcript):
    transcript_str = "\n".join([
        f"Q: {r['question']}\nA: {r['answer']} (Score: {r['score']}/10)" for r in transcript
    ])
    return f'''
You are a job candidate. Based on your resume and the following interview transcript, decide if you would accept an offer for this job. Respond with only YES or NO.

Resume:\n"""
{resume_text}
"""

Interview Transcript:\n"""
{transcript_str}
"""

Would you accept the offer? (YES or NO):
'''.strip()

import google.generativeai as genai
import json
import os
import random
from uuid import uuid4
from dotenv import load_dotenv

# --- INIT GEMINI ---
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")


# --- PROMPT GENERATORS ---

def generate_hiring_manager_prompt(jd_text):
    return f"""
You are a highly experienced hiring manager. Based on the job description below, ask one realistic and role-relevant interview question at a time. Avoid repeating questions.

Job Description:
\"\"\"
{jd_text}
\"\"\"
Ask the candidate one crisp interview question now:
""".strip()


def generate_candidate_prompt(resume_text, question):
    return f"""
You are a job candidate preparing for an interview. Based on your resume, answer the interview question in 3–5 lines. Be thoughtful, concise, and relevant.

Resume:
\"\"\"
{resume_text}
\"\"\"

Interview Question:
\"\"\"
{question}
\"\"\"

Your Answer:
""".strip()


def generate_eval_prompt(jd_text, answer):
    return f"""
You are a hiring manager evaluating a candidate’s interview answer based on the job description below.

Job Description:
\"\"\"
{jd_text}
\"\"\"

Candidate Answer:
\"\"\"
{answer}
\"\"\"

On a scale of 1 to 10, how strong is this answer in terms of relevance, clarity, and impact? Just respond with a single integer between 1 and 10.
""".strip()


# --- MAIN LOOP ---

def run_simulation(jd_text, resume_text, num_rounds=5):
    transcript = []
    total_score = 0
    interviewer_decisions = []
    candidate_decisions = []

    for i in range(1, num_rounds + 1):
        # Step 1: Hiring Manager asks question
        q_prompt = generate_hiring_manager_prompt(jd_text)
        question = model.generate_content(q_prompt).text.strip()

        # Step 2: Candidate answers
        a_prompt = generate_candidate_prompt(resume_text, question)
        answer = model.generate_content(a_prompt).text.strip()

        # Step 3: Score the answer
        s_prompt = generate_eval_prompt(jd_text, answer)
        score_raw = model.generate_content(s_prompt).text.strip()
        try:
            score = int(score_raw.split()[0])  # clean noisy outputs
            score = max(1, min(10, score))     # force score into 1–10
        except:
            score = random.randint(5, 8)       # fallback to midpoint if parse fails

        total_score += score

        # Per-round decisions
        round_transcript = transcript + [{"round": i, "question": question, "answer": answer, "score": score}]
        interviewer_prompt = generate_interviewer_decision_prompt(jd_text, round_transcript)
        interviewer_decision_raw = model.generate_content(interviewer_prompt).text.strip().upper()
        interviewer_decision = 'YES' in interviewer_decision_raw
        interviewer_decisions.append(interviewer_decision)

        candidate_prompt = generate_candidate_decision_prompt(resume_text, round_transcript)
        candidate_decision_raw = model.generate_content(candidate_prompt).text.strip().upper()
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

    # Final decision: average of per-round decisions
    avg_score = total_score / num_rounds
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
        "transcript_json": json.dumps(transcript, indent=2)
    }

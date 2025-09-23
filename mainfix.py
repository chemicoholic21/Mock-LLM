import streamlit as st
import json
import asyncio
import time

# ---- Replace/adjust this import to match your environment ----
# This is a placeholder for your actual model import.
# For example, it could be `from your_llm_library import model`
from simulator import model
# --------------------------------------------------------------

st.set_page_config(page_title="AI Avatar Hiring Simulator", layout="wide")
st.title("🧠 AI-to-AI Hiring Simulator")
st.markdown("""
Optimized *and* robust: batch + async for speed, plus validation & repair for accuracy.
""")

jd_text = st.text_area("📄 Job Description (JD)", height=220, placeholder="Paste the JD here...")
resume_text = st.text_area("👤 Candidate Resume", height=220, placeholder="Paste the Resume here...")
num_rounds = st.number_input("Number of rounds", min_value=1, max_value=500, value=5, step=1)
batch_size = st.number_input("Batch size (rounds per API call)", min_value=1, max_value=100, value=5, step=1)
st.caption("If results are incorrect, lower the batch size (e.g., 1-5) for higher fidelity.")

def _find_json_array(s: str):
    """
    Find the first balanced JSON array in `s` and return its substring, or None.
    Scans for first '[' and finds matching closing ']'.
    """
    if not s:
        return None
    start = s.find('[')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '[':
            depth += 1
        elif s[i] == ']':
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def _validate_round_obj(obj):
    """Return (is_valid, cleaned_obj_or_error_str)."""
    try:
        r = {}
        # round
        if 'round' not in obj:
            return False, "missing 'round'"
        rnum = int(obj['round'])
        r['round'] = rnum

        # question / answer
        q = str(obj.get('question', '')).strip()
        a = str(obj.get('answer', '')).strip()
        if not q: return False, f"round {rnum}: empty question"
        if not a: return False, f"round {rnum}: empty answer"
        r['question'] = q
        r['answer'] = a

        # score
        sc = obj.get('score', None)
        if sc is None:
            return False, f"round {rnum}: missing score"
        sc_int = int(sc)
        sc_int = max(1, min(10, sc_int))
        r['score'] = sc_int

        # decisions
        idc = str(obj.get('interviewer_decision', '')).strip().upper()
        cdc = str(obj.get('candidate_decision', '')).strip().upper()
        if idc not in ('YES', 'NO'):
            return False, f"round {rnum}: invalid interviewer_decision '{idc}'"
        if cdc not in ('YES', 'NO'):
            return False, f"round {rnum}: invalid candidate_decision '{cdc}'"
        r['interviewer_decision'] = idc
        r['candidate_decision'] = cdc

        return True, r
    except Exception as e:
        return False, f"validation error: {str(e)}"

# -------------------------
# Model wrapper (async-friendly)
# -------------------------
async def model_generate_text(prompt: str, *, retries=2, sleep_between=0.6):
    """
    Unified async wrapper:
    - If model has `generate_content_async`, await it.
    - Otherwise call `generate_content` in a thread via asyncio.to_thread.
    Tries to pass temperature=0.0 (deterministic) but will fall back if the method doesn't accept it.
    Returns the raw text response (string) or raises.
    """
    # Try the async API first (if present)
    gen_async = getattr(model, "generate_content_async", None)
    gen_sync = getattr(model, "generate_content", None)

    last_exc = None
    for attempt in range(retries):
        try:
            if gen_async is not None:
                # try calling with temperature=0.0, fall back if TypeError
                try:
                    resp = await gen_async(prompt, temperature=0.0)
                except TypeError:
                    resp = await gen_async(prompt)
                # expecting resp.text as in your original code
                return getattr(resp, "text", str(resp))
            elif gen_sync is not None:
                # run sync in thread
                def call_sync():
                    try:
                        r = gen_sync(prompt, temperature=0.0)
                    except TypeError:
                        r = gen_sync(prompt)
                    return getattr(r, "text", str(r))
                return await asyncio.to_thread(call_sync)
            else:
                raise RuntimeError("No compatible model.generate function found (generate_content_async or generate_content).")
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(sleep_between)
            else:
                raise last_exc
    raise last_exc

# -------------------------
# Simulation primitives
# -------------------------
EXAMPLE_JSON = [
    {
        "round": 1,
        "question": "What's your experience with X?",
        "answer": "I have 3 years doing X...",
        "score": 7,
        "interviewer_decision": "YES",
        "candidate_decision": "YES"
    }
]

async def simulate_batch_async(jd, resume, start_round, end_round, *, max_repair_attempts=2):
    """
    Ask the model to simulate rounds [start_round..end_round] in a single call and return
    a validated list of round dicts, or an {'error': '...'} dict.
    """
    # compact (but explicit) instruction + small example + strict JSON requirement
    prompt = f"""You will simulate an AI hiring conversation between a Hiring Manager and a Candidate.

Job Description:
{jd}

Candidate Resume:
{resume}

Simulate interview rounds {start_round} to {end_round} (inclusive). For each round produce:
- round (integer)
- question (string)
- answer (string)
- score (integer 1-10)
- interviewer_decision ("YES" or "NO")
- candidate_decision ("YES" or "NO")

IMPORTANT:
- **Return ONLY a single valid JSON array** (no explanation, no text before/after).
- The array must contain one object per round, with the exact keys above.
- Example output (format only):
{json.dumps(EXAMPLE_JSON, indent=2)}

Now produce the JSON array for the requested rounds.
"""

    try:
        raw = await model_generate_text(prompt)
    except Exception as e:
        return {"error": f"batch {start_round}-{end_round} model error: {str(e)}"}

    # Try to extract JSON array from raw text
    json_slice = _find_json_array(raw)
    parsed = None
    if json_slice:
        try:
            parsed = json.loads(json_slice)
        except Exception:
            parsed = None

    # If parsing failed, try repair attempts (ask the model to return valid JSON only)
    repair_attempt = 0
    while parsed is None and repair_attempt < max_repair_attempts:
        repair_prompt = f"""The previous response failed to be parsed as valid JSON. Extract and return ONLY a valid JSON array with objects for rounds {start_round}-{end_round} with these exact keys: round, question, answer, score (1-10), interviewer_decision (YES/NO), candidate_decision (YES/NO).
Here is the original response:
{raw}

Return ONLY the corrected JSON array (no other text).
"""
        try:
            raw_repair = await model_generate_text(repair_prompt)
            json_slice = _find_json_array(raw_repair)
            if json_slice:
                try:
                    parsed = json.loads(json_slice)
                except Exception:
                    parsed = None
            raw = raw_repair  # use repaired raw for next iteration if needed
        except Exception as e:
            return {"error": f"batch {start_round}-{end_round} repair error: {str(e)}"}
        repair_attempt += 1

    # Final fallback: if still can't parse, fall back to per-round simulation (slower but accurate)
    if parsed is None:
        # fall back to per-round single simulations (concurrently) and return their results
        single_tasks = [simulate_single_round_async(jd, resume, r) for r in range(start_round, end_round + 1)]
        single_results = await asyncio.gather(*single_tasks)
        recovered = []
        for res in single_results:
            if isinstance(res, dict) and res.get('error'):
                return {"error": f"batch {start_round}-{end_round} fallback single-round error: {res['error']}"}
            recovered.append(res)
        # ensure ordering
        recovered.sort(key=lambda x: x['round'])
        return recovered

    # Validate parsed entries
    if not isinstance(parsed, list):
        return {"error": f"batch {start_round}-{end_round} returned JSON that is not an array"}

    validated = []
    for obj in parsed:
        ok, result = _validate_round_obj(obj)
        if not ok:
            # if any item invalid, attempt one repair using the parsed raw as context
            # but to keep code simple: fall back to per-round simulation for those rounds
            # Build list of problematic rounds
            return_msg = f"batch {start_round}-{end_round} validation failed: {result}"
            # fallback to per-round for the whole batch for correctness
            single_tasks = [simulate_single_round_async(jd, resume, r) for r in range(start_round, end_round + 1)]
            single_results = await asyncio.gather(*single_tasks)
            recovered = []
            for res in single_results:
                if isinstance(res, dict) and res.get('error'):
                    return {"error": f"batch {start_round}-{end_round} fallback single-round error: {res['error']}"}
                recovered.append(res)
            recovered.sort(key=lambda x: x['round'])
            return recovered

        validated.append(result)

    # Sort by round and return
    validated.sort(key=lambda x: x['round'])
    return validated

async def simulate_single_round_async(jd, resume, rnum, *, max_retries=2):
    """Simulate exactly one round and return a validated dict or {'error':...}."""
    prompt = f"""You are simulating one interview round (round {rnum}) between a Hiring Manager and a Candidate.

Job Description:
{jd}

Candidate Resume:
{resume}

Produce EXACTLY one JSON object (no surrounding array) like:
{{
  "round": {rnum},
  "question": "...",
  "answer": "...",
  "score": 1,
  "interviewer_decision": "YES" or "NO",
  "candidate_decision": "YES" or "NO"
}}

Return only the JSON object (no extra text).
"""
    attempt = 0
    raw = None
    while attempt < max_retries:
        try:
            raw = await model_generate_text(prompt)
        except Exception as e:
            return {"error": f"round {rnum} model error: {str(e)}"}
        # attempt to extract an object by wrapping a single-object into an array for parsing convenience
        slice_candidate = raw.strip()
        # If it starts with '{', try to parse as single object
        try:
            if slice_candidate.startswith('{'):
                obj = json.loads(slice_candidate)
                ok, result = _validate_round_obj(obj)
                if ok:
                    return result
            # else, try to extract first object inside any text
            json_arr_slice = _find_json_array('[' + slice_candidate + ']')
            if json_arr_slice:
                arr = json.loads(json_arr_slice)
                if isinstance(arr, list) and len(arr) == 1:
                    ok, result = _validate_round_obj(arr[0])
                    if ok:
                        return result
        except Exception:
            pass
        # not valid — try repair attempt
        repair_prompt = f"""The previous response failed to be valid JSON for round {rnum}. Extract and return ONLY a single valid JSON object with the keys: round, question, answer, score (1-10), interviewer_decision (YES/NO), candidate_decision (YES/NO).
Here is the original response:
{raw}
Return ONLY the corrected JSON object.
"""
        try:
            raw = await model_generate_text(repair_prompt)
        except Exception as e:
            return {"error": f"round {rnum} repair error: {str(e)}"}
        attempt += 1

    return {"error": f"round {rnum} failed to produce valid JSON after {max_retries} tries"}

# -------------------------
# Orchestration
# -------------------------
def run_async(coro):
    """
    Runs an async coroutine, managing the event loop correctly.
    This is important for environments like Streamlit that might have their own loop management.
    """
    try:
        # Check if there's a running loop
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        # If no loop is running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

async def run_simulation_async(jd, resume, num_rounds, batch_size):
    tasks = []
    start = 1
    while start <= num_rounds:
        end = min(start + batch_size - 1, num_rounds)
        tasks.append(simulate_batch_async(jd, resume, start, end))
        start = end + 1

    # Run batches concurrently
    batch_results = await asyncio.gather(*tasks)

    # Flatten and handle errors
    transcript = []
    errors = []
    for res in batch_results:
        if isinstance(res, dict) and res.get('error'):
            errors.append(res['error'])
        elif isinstance(res, list):
            transcript.extend(res)
        else:
            errors.append(f"unexpected batch result type: {type(res)}")

    # Ensure we have results for requested rounds, otherwise it's an error
    transcript.sort(key=lambda x: x['round'])
    # If some rounds are missing, report
    got_rounds = {r['round'] for r in transcript}
    missing = [i for i in range(1, num_rounds + 1) if i not in got_rounds]
    if missing:
        errors.append(f"missing rounds: {missing}")

    # Compute stats using available transcript entries
    total_score = sum(r['score'] for r in transcript) if transcript else 0
    actual_rounds = len(transcript)
    avg_score = total_score / actual_rounds if actual_rounds > 0 else 0
    interviewer_final = (sum(1 for r in transcript if r['interviewer_decision'] == 'YES') / actual_rounds) >= 0.5 if actual_rounds > 0 else False
    candidate_final = (sum(1 for r in transcript if r['candidate_decision'] == 'YES') / actual_rounds) >= 0.5 if actual_rounds > 0 else False
    feedback = ("Strong overall! You clearly demonstrated skills aligned with the JD."
                if avg_score >= 7.5 else
                "Some answers lacked depth or relevance. Consider tailoring your responses better to the role.")

    return {
        "transcript": transcript,
        "average_score": avg_score,
        "interviewer_decision": interviewer_final,
        "candidate_decision": candidate_final,
        "final_match": (interviewer_final and candidate_final),
        "final_feedback": feedback,
        "transcript_json": json.dumps(transcript, indent=2),
        "errors": errors
    }

# -------------------------
# Streamlit trigger
# -------------------------
if st.button("Run Robust Async Simulation") and jd_text and resume_text:
    progress_bar = st.progress(0.0, text="starting...")
    status = st.empty()
    status.info("Preparing batches...")

    # Run the async simulation (this will block the Streamlit thread until done)
    with st.spinner("Running batches (concurrently) — this may take a few seconds..."):
        try:
            # FIX: Use the 'run_async' helper instead of 'asyncio.run' to avoid event loop conflicts.
            results = run_async(run_simulation_async(jd_text, resume_text, int(num_rounds), int(batch_size)))
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            # Stop execution here if a critical error occurs.
            st.stop()
            
    # Display results
    if results.get("errors"):
        st.warning("Some errors occurred (see details). Lower the batch size to improve fidelity if needed.")
        for e in results["errors"]:
            st.text(f"- {e}")

    st.success("✅ Interview Simulation Complete!")
    
    if results['transcript']:
        st.markdown("## 🗣️ Interview Exchange")
        for r in results['transcript']:
            st.markdown(f"""
**Round {r['round']}**
- **Q:** {r['question']}
- **A:** {r['answer']}
- **Score:** {r['score']}/10
- **Interviewer Decision:** {'YES ✅' if r['interviewer_decision'] == 'YES' else 'NO ❌'}
- **Candidate Decision:** {'YES ✅' if r['candidate_decision'] == 'YES' else 'NO ❌'}
""")

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
        st.download_button("📥 Download Transcript (JSON)", results["transcript_json"], file_name="transcript.json")
    else:
        st.error("Simulation completed but produced no results. Please check the errors above.")
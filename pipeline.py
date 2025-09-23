import json
import asyncio
import pandas as pd
import os
from dotenv import load_dotenv
import nest_asyncio

# --- Apply the patch to allow nested event loops ---
# This is the key fix for the "attached to a different loop" error.
nest_asyncio.apply()
# ----------------------------------------------------

# --- Load environment variables and configure the Gemini model ---
import google.generativeai as genai

# Load variables from .env file
load_dotenv()

# Configure the Gemini API with your key
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file or environment variables.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit()

# This class wrapper ensures compatibility with the existing script structure
class GeminiModel:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)

    async def generate_content_async(self, prompt: str, temperature=0.0):
        """Generates content asynchronously using the Gemini model."""
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        response = await self.model.generate_content_async(
            prompt,
            generation_config=generation_config
        )
        return response

# This is the 'model' object our script will use for the simulation
model = GeminiModel()
# ----------------------------------------------------------------

# -------------------------
# Utility functions
# -------------------------
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
async def model_generate_text(prompt: str, *, retries=3, sleep_between=1.0):
    """
    Unified async wrapper that calls the Gemini model and returns the text response.
    """
    gen_async = getattr(model, "generate_content_async", None)
    if not gen_async:
        raise RuntimeError("The provided model object does not have a 'generate_content_async' method.")

    last_exc = None
    for attempt in range(retries):
        try:
            # Await the async generation method
            resp = await gen_async(prompt, temperature=0.0)
            # The Gemini response object has a '.text' attribute
            return getattr(resp, "text", str(resp))
        except Exception as e:
            print(f"Model generation error on attempt {attempt + 1}: {e}")
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(sleep_between)
            else:
                # Re-raise the last exception if all retries fail
                raise last_exc
    # This line should not be reachable if retries > 0
    raise last_exc if last_exc else RuntimeError("Model generation failed without a specific exception.")


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

    json_slice = _find_json_array(raw)
    parsed = None
    if json_slice:
        try:
            parsed = json.loads(json_slice)
        except Exception:
            parsed = None

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
            raw = raw_repair
        except Exception as e:
            return {"error": f"batch {start_round}-{end_round} repair error: {str(e)}"}
        repair_attempt += 1

    if parsed is None:
        return {"error": f"batch {start_round}-{end_round} failed to produce valid JSON after repairs."}


    if not isinstance(parsed, list):
        return {"error": f"batch {start_round}-{end_round} returned JSON that is not an array"}

    validated = []
    has_errors = False
    for obj in parsed:
        ok, result = _validate_round_obj(obj)
        if not ok:
            has_errors = True
            break
        validated.append(result)

    if has_errors:
        return {"error": f"batch {start_round}-{end_round} failed validation."}


    validated.sort(key=lambda x: x['round'])
    return validated

# -------------------------
# Orchestration
# -------------------------
def run_async(coro):
    """
    Runs an async coroutine, managing the event loop correctly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
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

    batch_results = await asyncio.gather(*tasks)

    transcript = []
    errors = []
    for res in batch_results:
        if isinstance(res, dict) and res.get('error'):
            errors.append(res['error'])
        elif isinstance(res, list):
            transcript.extend(res)
        else:
            errors.append(f"unexpected batch result type: {type(res)}")

    transcript.sort(key=lambda x: x['round'])
    got_rounds = {r['round'] for r in transcript}
    missing = [i for i in range(1, num_rounds + 1) if i not in got_rounds]
    if missing:
        errors.append(f"missing rounds: {missing}")

    actual_rounds = len(transcript)
    interviewer_yes_count = sum(1 for r in transcript if r['interviewer_decision'] == 'YES')
    candidate_yes_count = sum(1 for r in transcript if r['candidate_decision'] == 'YES')
    
    interviewer_final = (interviewer_yes_count / actual_rounds) >= 0.5 if actual_rounds > 0 else False
    candidate_final = (candidate_yes_count / actual_rounds) >= 0.5 if actual_rounds > 0 else False
    
    return {
        "final_match": (interviewer_final and candidate_final),
        "errors": errors
    }

def get_llm_evaluation(jd_text, resume_text):
    """
    This function takes the Job Description and Resume text,
    runs the simulation, and returns 'yes' or 'no'.
    """
    num_rounds = 5
    batch_size = 5
    
    try:
        results = run_async(run_simulation_async(jd_text, resume_text, int(num_rounds), int(batch_size)))
        if results.get("errors"):
            print("Errors encountered during simulation:", results["errors"])
        
        return "yes" if results.get("final_match") else "no"
        
    except Exception as e:
        print(f"An error occurred during simulation: {e}")
        return "no"

# -------------------------
# Main execution block
# -------------------------
if __name__ == "__main__":
    input_filename = 'Eval 23rd Sept - test.csv'
    try:
        df = pd.read_csv(input_filename)
        
        df['mock llm'] = ''
        
        for index, row in df.iterrows():
            # Use the correct column names from your CSV
            jd = row['Grapevine Job - Job → Description']
            resume = row['Grapevine Userresume - Resume → Metadata → Resume Text']
            
            # Skip rows where either JD or resume is missing to avoid errors
            if pd.isna(jd) or pd.isna(resume):
                print(f"Skipping row {index + 1} due to missing data.")
                df.at[index, 'mock llm'] = 'error - missing data'
                continue

            print(f"Processing row {index + 1}/{len(df)}...")
            
            decision = get_llm_evaluation(jd, resume)
            
            df.at[index, 'mock llm'] = decision

        output_filename = 'evaluation_results.csv'
        df.to_csv(output_filename, index=False)
        print(f"\nProcessing complete. Results saved to {output_filename}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except KeyError as e:
        print(f"Error: A required column was not found in the CSV: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
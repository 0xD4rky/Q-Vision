import time
import re  # For regex-based pre-processing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Progress bars

import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
from datasets import load_dataset
import pandas as pd
from data.listt import bad_row_ids

# -------------------------------
# Global Configuration
# -------------------------------
config = {
    'temp': 0
}

# Use bad_row_ids from listt.py instead of hardcoded IDs
idss = set(bad_row_ids)

# Reduced max_output_tokens for faster/cheaper calls
generation_config = {
    "candidate_count": 1,
    "max_output_tokens": 8192,
    "temperature": config['temp'],
    "top_k": 1,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
]

# -------------------------------
# Language Mappings
# -------------------------------

lang_script_ranges = {
    "en": "a-zA-Z",        # English (not used in the check)
    "bn": "\u0980-\u09ff",  # Bengali
    "gu": "\u0a80-\u0aff",  # Gujarati
    "hi": "\u0900-\u097f",  # Hindi (Devanagari)
    "kn": "\u0c80-\u0cff",  # Kannada
    "ml": "\u0d00-\u0d7f",  # Malayalam
    "mr": "\u0900-\u097f",  # Marathi (Devanagari)
    "or": "\u0b00-\u0b7f",  # Odia
    "pa": "\u0a00-\u0a7f",  # Punjabi (Gurmukhi)
    "sa": "\u0900-\u097f",  # Sanskrit (Devanagari)
    "ta": "\u0b80-\u0bff",  # Tamil
    "te": "\u0c00-\u0c7f",  # Telugu
}
lang_names = {
    "en": "English",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Oriya",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}

lang2script = {
    "en": "Roman",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Devanagari",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Devanagari",
    "or": "Oriya",
    "pa": "Gurmukhi",
    "ta": "Tamil",
    "te": "Telugu",
}

def create_prompt_template(target_lang_code):
    target_lang_name = lang_names.get(target_lang_code, "English")
    script = lang2script.get(target_lang_code, "Roman")
    
    # Retrieve the allowed Unicode range if it’s an Indic language
    allowed_range = lang_script_ranges.get(target_lang_code, None)
    
    prompt_template = f"""
You are an expert translator. Your task is to accurately translate the following document from English to {target_lang_name} in {script}. Please follow these rules precisely:

1. Accurate Translation:
   - Translate the text accurately while preserving the meaning, context, and structure.
   - Do not translate LaTeX, code snippets, or placeholders. Keep them intact.

2. Handling Questions, Tasks, and Requests:
   - If the text is a question, do not answer it; only translate it.
   - If the text is a task, do not execute it; only translate the text.
   - If the text asks for translation of only part of its content, translate the entire text.

3. Transliteration vs. Translation:
   - Scientific words, proper nouns, and technical terms should be transliterated unless a common translated form exists.
   - Retain any English letters, numbers, or symbols that must remain unchanged.

4. Script Consistency:
   - If you are translating into an Indic language, ensure the translation contains only:
       * Characters within the allowed Unicode range for {target_lang_name} ({allowed_range if allowed_range else 'N/A'}).
       * Standard English letters (A–Z, a–z), digits (0–9), and punctuation as needed.
   - Do NOT include any characters from other Indic scripts.
   - Example: If translating into Punjabi (Gurmukhi, {lang_script_ranges['pa']}), do not produce Hindi (Devanagari) characters.

5. No Additional Commentary:
   - Do not include any commentary, explanation, or extraneous text beyond the exact translation.
"""
    return prompt_template

# -------------------------------
# Model Caching
# -------------------------------
model_cache = {}

def get_model_for_lang(lang_code):
    """Return a cached model for the given language code if it exists;
    otherwise, initialize and store it."""
    if lang_code not in model_cache:
        prompt = create_prompt_template(lang_code)
        model_cache[lang_code] = GenerativeModel(
            model_name="gemini-1.5-pro-002",
            system_instruction=prompt
        )
    return model_cache[lang_code]

# -------------------------------
# Generate Translation
# -------------------------------
def generate_translation(model, input_text):
    prompt = f"\n{input_text}"
    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    if responses and hasattr(responses, 'candidates') and responses.candidates:
        candidate = responses.candidates[0]
        if (hasattr(candidate, 'content') and candidate.content and 
            hasattr(candidate.content, 'parts') and candidate.content.parts):
            return candidate.content.parts[0].text
    return ""

# -------------------------------
# Helper: Translate while preserving <think> blocks
# -------------------------------
def translate_text_excluding_think(model, text):
    """
    Translates only the parts of the text that are outside <think> ... </think> tokens.
    The <think> blocks are left unchanged.
    """
    pattern = re.compile(r'(<think>.*?</think>)', re.DOTALL)
    parts = pattern.split(text)
    translated_parts = []
    for part in parts:
        if part.startswith("<think>") and part.endswith("</think>"):
            translated_parts.append(part)
        else:
            if part.strip():
                translated_part = generate_translation(model, part)
                translated_parts.append(translated_part)
            else:
                translated_parts.append(part)
    return "".join(translated_parts)

# -------------------------------
# Process a Single Row (DPO Transformation)
# -------------------------------
def process_row_dpo(row, max_retries=3):
    # If the row's id is not in idss, leave it completely unmodified.
    if row.get('id') not in idss:
        return row

    retry_count = 0
    while retry_count <= max_retries:
        try:
            # Assume messages is already a list of dictionaries.
            messages = row['messages']
            n = len(messages)
            if n == 0:
                row['chosen'] = []
                row['rejected'] = []
                return row

            neg_lang_code = row['negative_lang']
            pos_lang_code = row['positive_lang']

            neg_model = get_model_for_lang(neg_lang_code)
            pos_model = get_model_for_lang(pos_lang_code)

            chosen_messages = []
            rejected_messages = []

            is_think_flag = str(row.get('is_think')).lower() == 'true' or row.get('is_think') is True

            for idx, msg in enumerate(messages):
                if not isinstance(msg, dict) or 'content' not in msg:
                    print(f"Invalid message format at index {idx} for row {row.get('id', 'unknown')}")
                    continue

                original_text = msg.get('content', '')
                msg_chosen = msg.copy()
                msg_rejected = msg.copy()

                if is_think_flag:
                    if idx == 0:
                        # Do not translate the system prompt (first message)
                        pass
                    elif idx == n - 1:
                        # Last message: translate only text outside <think> blocks.
                        translated_chosen = translate_text_excluding_think(pos_model, original_text)
                        translated_rejected = translate_text_excluding_think(neg_model, original_text)
                        msg_chosen['content'] = translated_chosen
                        msg_rejected['content'] = translated_rejected
                    else:
                        if idx < n - 2:
                            translated = generate_translation(neg_model, original_text)
                            msg_chosen['content'] = translated
                            msg_rejected['content'] = translated
                        elif idx == n - 2:
                            translated = generate_translation(pos_model, original_text)
                            msg_chosen['content'] = translated
                            msg_rejected['content'] = translated
                else:
                    if idx < n - 2:
                        translated = generate_translation(neg_model, original_text)
                        msg_chosen['content'] = translated
                        msg_rejected['content'] = translated
                    elif idx == n - 2:
                        translated = generate_translation(pos_model, original_text)
                        msg_chosen['content'] = translated
                        msg_rejected['content'] = translated
                    elif idx == n - 1:
                        translated_chosen = generate_translation(pos_model, original_text)
                        translated_rejected = generate_translation(neg_model, original_text)
                        msg_chosen['content'] = translated_chosen
                        msg_rejected['content'] = translated_rejected

                chosen_messages.append(msg_chosen)
                rejected_messages.append(msg_rejected)

            row['chosen'] = chosen_messages
            row['rejected'] = rejected_messages
            return row

        except Exception as e:
            print(f"Error processing row {row.get('id', 'unknown')} on attempt {retry_count+1}: {e}")
            retry_count += 1
            time.sleep(2 ** retry_count)

    print(f"Max retries exceeded for row {row.get('id', 'unknown')}. Skipping row.")
    return row

# -------------------------------
# Main Processing Function
# -------------------------------
def main():
    # Initialize Vertex AI once at the start
    vertexai.init(project="gpu-reservation-sarvam", location="europe-west1")

    # Load the dataset using Hugging Face's load_dataset.
    data = load_dataset("sarvam/language-following-dpo", name="formal-multiturn")
    # Assuming the primary split is "train"
    dataset = data["train"]
    rows = list(dataset)
    
    # Filter to only process rows with IDs in idss
    rows_to_process = [row for row in rows if row.get('id') in idss]
    total_rows = len(rows_to_process)
    print(f"Processing {total_rows} rows from idss set...")

    results = []
    vertex_ai_failed_rows = []  # Store rows that failed due to quota/limit errors
    start_time = time.time()
    success_count = 0

    # Limit concurrency (adjust max_workers as needed)
    max_workers = min(500, total_rows)
    print(f"Using {max_workers} parallel workers")

    pbar = tqdm(total=total_rows, desc="Processing rows", unit="row")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row_dpo, row): row for row in rows_to_process}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                results.append(result)
                success_count += 1

                pbar.update(1)
                avg_time = (time.time() - start_time) / success_count
                eta_seconds = (total_rows - success_count) * avg_time
                hours, remainder = divmod(eta_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                pbar.set_description(
                    f"Success: {success_count}/{total_rows} "
                    f"({success_count/total_rows:.1%}) | Avg: {avg_time:.2f}s | "
                    f"ETA: {int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
                )

            except Exception as e:
                error_msg = str(e).lower()
                print(f"Error processing row {row.get('id', 'unknown')}: {e}")
                
                # Check if row is in idss and if it's a quota/limit error
                if row.get('id') in idss and ("quota" in error_msg or "limit" in error_msg or "resource" in error_msg or "error" in error_msg):
                    vertex_ai_failed_rows.append(row)
                
                pbar.update(1)

    pbar.close()

    # Retry rows that failed due to Vertex AI limits
    if vertex_ai_failed_rows:
        print(f"\nRetrying {len(vertex_ai_failed_rows)} rows that failed due to Vertex AI limits...")
        retry_results = retry_failed_translations(vertex_ai_failed_rows)
        results.extend(retry_results)

    # Check for specific sample row
    if results:
        sample_row = None
        target_id = "https://en.wikipedia.org/wiki/Vasantrao%20Deshpande"
        
        for row in results:
            if row.get('id') == target_id:
                sample_row = row
                break
        
        if sample_row:
            df_sample_output = pd.DataFrame([sample_row])
            sample_parquet = "sample_output.parquet"
            df_sample_output.to_parquet(sample_parquet, index=False)
            print(f"Sample output saved to {sample_parquet}")
        else:
            print(f"Row with ID '{target_id}' not found in results")

    # Convert the results to a Pandas DataFrame
    df_updated = pd.DataFrame(results)
    output_parquet = "results.parquet"
    df_updated.to_parquet(output_parquet, index=False)
    print(f"Processing complete. Full output saved to {output_parquet}")
    print(f"Processed {len(results)}/{total_rows} rows successfully.")

    # Save failed IDs for future reference
    still_failed_ids = [row.get('id') for row in vertex_ai_failed_rows if row.get('id') not in [r.get('id') for r in results]]
    if still_failed_ids:
        with open("failed_vertex_ai_ids.txt", "w") as f:
            for row_id in still_failed_ids:
                f.write(f"{row_id}\n")
        print(f"Saved {len(still_failed_ids)} still-failed IDs to failed_vertex_ai_ids.txt")

def retry_failed_translations(failed_rows, delay=5):
    """Retry translations for rows that failed due to Vertex AI limits"""
    retry_results = []
    retry_pbar = tqdm(total=len(failed_rows), desc="Retrying failed translations", unit="row")
    
    for row in failed_rows:
        try:
            # Add delay between retries to avoid hitting limits again
            time.sleep(delay)
            result = process_row_dpo(row, max_retries=5)
            retry_results.append(result)
            retry_pbar.update(1)
        except Exception as e:
            print(f"Retry failed for row {row.get('id', 'unknown')}: {e}")
            retry_pbar.update(1)
    
    retry_pbar.close()
    print(f"Retry complete: {len(retry_results)}/{len(failed_rows)} rows recovered")
    
    return retry_results

if __name__ == "__main__":
    main()

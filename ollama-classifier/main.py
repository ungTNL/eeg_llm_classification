import pandas as pd
import instructor
from openai import AsyncOpenAI
import asyncio
import aiosqlite
import sqlite3
import time
from config import *
from prompting import system_prompt

# 1. Initialize the Async Instructor client
llm = instructor.from_openai(
        AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama", # api_key is ignored by Ollama
    ),
    mode=instructor.Mode.JSON,
)

# 3. Database setup function
async def setup_db():
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS extractions (
                note_id INTEGER PRIMARY KEY,
                study_number TEXT,
                emu_label TEXT,
                
                diff_enceph_val BOOLEAN,
                diff_enceph_text TEXT,
                diff_enceph_mod TEXT,
                
                sz_epileptic_val BOOLEAN,
                sz_epileptic_text TEXT,
                sz_non_epileptic_val BOOLEAN,
                sz_non_epileptic_text TEXT,
                sz_status_epilepticus_val BOOLEAN,
                sz_status_epilepticus_text TEXT,
                sz_ncse_val BOOLEAN,
                sz_ncse_text TEXT,
                
                ed_val BOOLEAN,
                ed_text TEXT,
                ed_type TEXT,
                
                asym_val BOOLEAN,
                asym_text TEXT,
                iic_val BOOLEAN,
                iic_text TEXT,
                gpd_val BOOLEAN,
                gpd_text TEXT,
                gpdt_val BOOLEAN,
                gpdt_text TEXT,
                lpd_val BOOLEAN,
                lpd_text TEXT,
                focal_slowing_val BOOLEAN,
                focal_slowing_text TEXT,
                
                inference_time REAL,
                error TEXT
            )
        """)
        await db.commit()

# 4. The core extraction worker
async def process_note(note_id: int, note_text: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        async with aiosqlite.connect(DB_FILE) as db:
            async with db.execute("SELECT 1 FROM extractions WHERE note_id = ?", (note_id,)) as cursor:
                if await cursor.fetchone():
                    print(f"Skipping {note_id} - already processed.")
                    return

        print(f"Processing note {note_id}...")
        start_time = time.time()
        
        try:
            extraction = await llm.chat.completions.create(
                model="llama3.1", 
                messages=[
                    {"role": "system", "content": "Extract the requested fields. Ensure exact quotes are used for evidence text."},
                    {"role": "user", "content": note_text}
                ],
                response_model=EEGExtraction,
                max_retries=3, 
                extra_body={"options": {"num_ctx": 8192}} 
            )
            
            inference_time = time.time() - start_time
            
            # Join the list of Enums into a single string safely
            ed_types_joined = ", ".join([t.value for t in extraction.epileptiform_discharges_type]) if extraction.epileptiform_discharges_type else ""
            
            # Massive Flattened Insertion
            async with aiosqlite.connect(DB_FILE) as db:
                await db.execute("""
                    INSERT INTO extractions (
                        note_id, study_number, emu_label,
                        diff_enceph_val, diff_enceph_text, diff_enceph_mod,
                        sz_epileptic_val, sz_epileptic_text, sz_non_epileptic_val, sz_non_epileptic_text, 
                        sz_status_epilepticus_val, sz_status_epilepticus_text, sz_ncse_val, sz_ncse_text,
                        ed_val, ed_text, ed_type,
                        asym_val, asym_text, iic_val, iic_text, gpd_val, gpd_text,
                        gpdt_val, gpdt_text, lpd_val, lpd_text, focal_slowing_val, focal_slowing_text,
                        inference_time
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    note_id, 
                    extraction.study_number, 
                    extraction.emu_label.value if extraction.emu_label else None,
                    
                    extraction.diffuse_encephalopathy.value,
                    extraction.diffuse_encephalopathy.text,
                    extraction.diffuse_encephalopathy_modifier.value if extraction.diffuse_encephalopathy_modifier else None,
                    
                    extraction.sz_epileptic.value,
                    extraction.sz_epileptic.text,
                    extraction.sz_non_epileptic.value, 
                    extraction.sz_non_epileptic.text, 
                    extraction.sz_status_epilepticus.value,
                    extraction.sz_status_epilepticus.text,
                    extraction.sz_nonconvulsive_status_epilepticus.value,
                    extraction.sz_nonconvulsive_status_epilepticus.text,
                    
                    extraction.epileptiform_discharges.value,
                    extraction.epileptiform_discharges.text,
                    ed_types_joined,
                    
                    extraction.asymmetric.value, extraction.asymmetric.text,
                    extraction.ictal_interictal_cont.value, extraction.ictal_interictal_cont.text,
                    extraction.gen_pd.value, extraction.gen_pd.text,
                    extraction.gen_pdt.value, extraction.gen_pdt.text,
                    extraction.lat_pd.value, extraction.lat_pd.text,
                    extraction.focal_slowing.value, extraction.focal_slowing.text,
                    
                    inference_time
                ))
                await db.commit()
                
        except Exception as e:
            inference_time = time.time() - start_time
            print(f"Error on {note_id}: {e}")
            
            async with aiosqlite.connect(DB_FILE) as db:
                await db.execute("INSERT INTO extractions (note_id, error, inference_time) VALUES (?, ?, ?)", 
                                 (note_id, str(e), inference_time))
                await db.commit()

# 5. The Batch Manager
async def main():
    await setup_db()
    
    df = pd.read_excel(INPUT_FILE)
    if NUM_TEST:
        df = df.head(NUM_TEST)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [process_note(row['note_id'], row['note_text'], semaphore) for index, row in df.iterrows()]
    
    await asyncio.gather(*tasks)
    print("All inferences processed!")
    
    # Export to Excel
    print("Exporting merged results to Excel...")
    conn = sqlite3.connect(DB_FILE)
    results_df = pd.read_sql_query("SELECT * FROM extractions", conn)
    results_df['note_id'] = results_df['note_id'].astype(int)
    conn.close()
    
    merged_df = pd.merge(df, results_df, on="note_id", how="left")
    merged_df.to_excel(OUTPUT_FILE, index=False)
    print("Export complete! Saved as 'eeg_results_merged.xlsx'.")

if __name__ == "__main__":
    asyncio.run(main())
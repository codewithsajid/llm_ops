import os
import json
import mlflow
import tempfile
from typing import List, Dict

def log_run(
    question: str,
    chunk_ids: List[str],
    answer: str,
    stats: Dict[str, float],
    prompt: str,
    rerank_scores: List[List[float]],
    search_steps: List[Dict],
    citations: str = None
) -> str:
    """Log experiment run to MLflow with enhanced metadata."""
    
    with mlflow.start_run() as run:
        # Ensure all string data is properly encoded
        safe_question = question.encode('utf-8', errors='ignore').decode('utf-8')
        safe_answer = answer.encode('utf-8', errors='ignore').decode('utf-8')
        
        mlflow.log_params({
            "question": safe_question,
            "n_chunks": len(chunk_ids),
            "n_steps": len(search_steps)
        })
        
        mlflow.log_metrics(stats)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write files with explicit UTF-8 encoding
            for filename, content in [
                ("prompt.txt", prompt),
                ("answer.txt", safe_answer),
                ("citations.txt", citations or "")
            ]:
                if content:
                    filepath = os.path.join(tmp_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content.encode('utf-8', errors='ignore').decode('utf-8'))
                    mlflow.log_artifact(filepath)
            
            # Log search steps
            steps_path = os.path.join(tmp_dir, "search_steps.json")
            with open(steps_path, "w", encoding="utf-8") as f:
                json.dump(search_steps, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(steps_path)
        
        return run.info.run_id


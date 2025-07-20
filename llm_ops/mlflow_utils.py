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
    citations: str = None  # Added citations parameter
) -> str:
    """Log experiment run to MLflow with enhanced metadata."""
    
    with mlflow.start_run() as run:
        # Log basic metrics
        mlflow.log_params({
            "question": question,
            "n_chunks": len(chunk_ids),
            "n_steps": len(search_steps)
        })
        
        # Log performance metrics
        mlflow.log_metrics(stats)
        
        # Log artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Log prompt and answer
            prompt_path = os.path.join(tmp_dir, "prompt.txt")
            with open(prompt_path, "w") as f:
                f.write(prompt)
            mlflow.log_artifact(prompt_path)
            
            answer_path = os.path.join(tmp_dir, "answer.txt")
            with open(answer_path, "w") as f:
                f.write(answer)
            mlflow.log_artifact(answer_path)
            
            # Log citations if available
            if citations:
                citations_path = os.path.join(tmp_dir, "citations.txt")
                with open(citations_path, "w") as f:
                    f.write(citations)
                mlflow.log_artifact(citations_path)
            
            # Log search steps
            steps_path = os.path.join(tmp_dir, "search_steps.json")
            with open(steps_path, "w") as f:
                json.dump(search_steps, f, indent=2)
            mlflow.log_artifact(steps_path)
        
        return run.info.run_id

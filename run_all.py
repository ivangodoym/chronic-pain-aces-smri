"""
run_all.py
Master execution script for the entire chronic pain mediation pipeline.
"""
import sys 
import time 
from pathlib import Path 
from src.utils import find_input_file, print_title
from src.preprocessing import run_preprocessing 
from src.features import run_feature_engineering
from src.modeling import run_modeling
from src.sem_modeling import run_sem_pipeline 
from src.visualization import run_visualization 

def main() -> None: 
    start_total = time.time()
    print_title("CHRONIC PAIN ACES-sMRI PIPELINE: FULL REPRODUCIBLE RUN")
    try:
        input_file = find_input_file()
        print(f"Data file detected: {input_file}")
    except FileNotFoundError:
        print("\n [ERROR] Raw data file ACEs_Neuroimages.xlsx was not found.")
        print("Please place the Excel file in data/raw (or data/) and run again.")
        sys.exit(1)

    try: 
        run_preprocessing()
        run_feature_engineering()
        run_modeling()
        run_sem_pipeline()
        run_visualization()

    except Exception as e: 
        print(f"\n[ERROR] Pipeline aborted due to an error: {e}")
        sys.exit(1)

    elapsed = time.time() - start_total 
    print_title(f"PIPELINE COMPLETED SUCESSFULLY in {elapsed:.1f} seconds")
    print("Tables saved to : outputs/tables/")
    print("Figures saved to : outputs/figures/\n")

if __name__ == "__main__":
    main()
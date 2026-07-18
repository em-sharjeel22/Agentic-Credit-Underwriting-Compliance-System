from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "streamlit_app.py"

if not APP_PATH.exists():
    raise FileNotFoundError(f"Main Streamlit app not found at: {APP_PATH}")

sys.path.insert(0, str(PROJECT_ROOT))
runpy.run_path(str(APP_PATH), run_name="__main__")

import subprocess

import _bootstrap  # noqa: F401


if __name__ == "__main__":
    subprocess.run(["streamlit", "run", "ui/streamlit_app.py"], check=True)

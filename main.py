import os
# Configure environment variable at process start
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import subprocess
import sys

# Start Streamlit CLI as a subprocess inheriting the environment variable
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
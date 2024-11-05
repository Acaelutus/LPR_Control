import subprocess
import sys

if __name__ == "__main__":
    python_script = "final.py"
    subprocess.Popen([sys.executable, python_script])

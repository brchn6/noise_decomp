# Helper script to build the Cython extension
# Usage: python build_cython.py
import subprocess
subprocess.check_call([
    'python', 'setup_cython.py', 'build_ext', '--inplace'
])
print("Cython extension built.")

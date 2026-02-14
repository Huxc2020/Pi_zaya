
import sys
import traceback

try:
    print("Attempting to import kb.converter...")
    import kb.converter.runner
    print("Success!")
except Exception:
    traceback.print_exc()

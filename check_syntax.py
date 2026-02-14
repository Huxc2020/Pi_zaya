
import sys
import glob
import traceback

files = glob.glob("kb/converter/*.py")
print(f"Checking {len(files)} files...")

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fp:
            content = fp.read()
        compile(content, f, "exec")
        print(f"OK: {f}")
    except SyntaxError as e:
        print(f"FAIL: {f}")
        traceback.print_exc()
    except Exception as e:
        print(f"ERROR: {f} - {e}")

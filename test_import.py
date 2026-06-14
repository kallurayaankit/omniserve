import sys, traceback, os
os.chdir("C:\\Users\\user\\AIprojects\\omniserve\\backend")
sys.path.insert(0, "C:\\Users\\user\\AIprojects\\omniserve")
try:
    import main
    print("OK")
except Exception:
    traceback.print_exc()
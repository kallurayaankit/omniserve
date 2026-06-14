import os, sys, traceback, logging

# Log EVERYTHING to a file
logging.basicConfig(
    filename="backend_error.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ".")

    logging.info("Starting import of backend.main")
    import backend.main
    logging.info("Import successful")

    app = backend.main.app
    logging.info("Starting Uvicorn")

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
except Exception:
    traceback.print_exc()
    logging.error("Fatal error", exc_info=True)
    print("ERROR CAPTURED – check backend_error.log")
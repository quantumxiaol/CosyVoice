import argparse
import os
import sys
import uvicorn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice3 FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8891, help="Port to listen on")
    args = parser.parse_args()

    from fastapi_service.service import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=True)


if __name__ == "__main__":
    main()

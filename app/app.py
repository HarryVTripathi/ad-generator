import logging
from uvicorn import run
from fastapi import FastAPI
from controllers import ad_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Ad generator service",
        description="Generates in ad based on prompts",
    )

    @app.get("/")
    def root():
        return {"message": "app is live"}
    
    app.include_router(ad_router)

    return app


def run_server():
    run(
        "app:create_app",
        host="0.0.0.0",
        port=8084,
        factory=True,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    logging.info("App is running...")
    try:
        run_server()
    except Exception as e:
        logging.error(str(e))

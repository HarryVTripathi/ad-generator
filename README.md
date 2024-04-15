# ad-generator
Generating ad from text to image diffusion models

Built on Python 3.12

To run the service:

:rocket: Create a virtual environment

```shell 
python -m venv .
./Script/activate
```

:rocket: Install dependencies

```shell
pip install -r requirements-dev.txt
```

:rocket: Start uvicorn server

```shell
python .\app\app.py
```

:rocket: App will run on localhost, port 8084. it the `/docs` endpoint for Swagger UI.

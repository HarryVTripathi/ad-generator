import io
import traceback
import os
import json
from uuid import uuid4
from pathlib import Path
import numpy as np
from PIL import Image, ImageFont, ImageDraw 
from fastapi import APIRouter, BackgroundTasks, File, UploadFile, Response
from fastapi.responses import FileResponse

from generator.image_generator import generate_image
from models.sd_config import AdParams
from utils.constants import (
    default_negative_prompt,
    FILE_EXTENSION,
)


ad_router = APIRouter(tags=["ad generator"])


@ad_router.post("/ad")
async def generate_ad(ad_params: AdParams, background_task: BackgroundTasks):
    """
    Asynchronously generates an image.

    Response includes generation_id which can be used to fetch the image, once generated.
    """
    try:
        trace_id = str(uuid4())
        span_id = str(uuid4())

        prompt = ad_params.prompt
        negative_prompt = ad_params.negative_prompt or default_negative_prompt

        background_task.add_task(generate_image, trace_id, prompt, negative_prompt)

        response = {
            "generation_id": trace_id,
            "message": "Request has been queued",
        }

        return Response(content=json.dumps(response), status_code=200, media_type="application/json")
    
    except Exception as e:
        response = {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "Internal server error",
        }
        return Response(content=json.dumps(response), status_code=500, media_type="application/json")


@ad_router.get("/ad/{generation_id}")
def get_ad(generation_id: str):
    try:
        fp = r"" + str(Path(__file__).parents[1]) + "/" + generation_id + FILE_EXTENSION
        im = Image.open(fp)

        if im.mode in ("RGBA", "P"): 
            im = im.convert("RGB")

        img = im
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format=img.format)
        imgByteArr = imgByteArr.getvalue()

        return Response(content=imgByteArr, media_type="image/jpeg")
    
    except FileNotFoundError as f:
        response = {
            "code": "NOT_FOUND",
            "message": "File not found. Please try again after a while or try generating a new ad."
        }
        return Response(content=json.dumps(response), status_code=404, media_type="application/json")

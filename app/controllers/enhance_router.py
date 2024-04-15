import io
import logging
import traceback
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Response
from fastapi.responses import FileResponse
from generator.enhance_generator import filhance


enhance_router = APIRouter(tags=["enhance", "filter"])

@enhance_router.post("/enhance")
def filter_enhance_products(file: UploadFile = File()):
    """
    file: Expected to be an image with apparel products
    """
    try:
        im = Image.open(file.file)
        if im.mode in ("RGBA", "P"): 
            im = im.convert("RGB")

        img = filhance(im)
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format="JPEG")
        imgByteArr = imgByteArr.getvalue()

        return Response(content=imgByteArr, media_type="image/jpeg")

    except Exception as e:
        logging.error(traceback.format_exc())
        return Response(status_code=400)
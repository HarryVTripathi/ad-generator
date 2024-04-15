from PIL import Image, ImageFont, ImageDraw 
from fastapi import APIRouter, File, UploadFile, Response
from fastapi.responses import FileResponse


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


    except:
        pass
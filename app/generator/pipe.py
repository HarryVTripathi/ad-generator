from pathlib import Path
from typing import Unpack
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from generator.singleton import SingletonMeta
from models.sd_config import BasePipeConfig
from utils.exceptions import NSFWGeneratedError
from utils.constants import (
    DEVICE,
    FILE_EXTENSION,
    MODEL_PATH,
)
   

class SDPipe(metaclass=SingletonMeta):
    def __init__(self, model_path=None) -> None:
        print("init called")
        self.model_path = model_path or MODEL_PATH
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
        )
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(self._pipe.scheduler.config)
        self._pipe = self._pipe.to(DEVICE)


    @property
    def get_pipe(self):
        return self._pipe
    

    def generate(self, generation_id: str, **kwargs: Unpack[BasePipeConfig]):
        print(f"kwargs: {kwargs}")
        pipe = self._pipe
        imgs_dict = pipe(**kwargs)

        if not imgs_dict.nsfw_content_detected[0]:
            image: Image = imgs_dict.images[0]
            file_path = r"" + str(Path(__file__).parents[1]) + "/" + generation_id + FILE_EXTENSION
            image.save(file_path)

            return "Successfully Generated."

        else:
            raise NSFWGeneratedError()

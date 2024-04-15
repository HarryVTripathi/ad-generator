from typing import TypedDict, Optional
from pydantic import BaseModel


class BasePipeConfig(TypedDict):
    prompt: str
    negative_prompt: str
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    height: int = 512
    width: int = 512
    clip_skip: int = 1
    num_images_per_prompt: int = 1


class AdParams(BaseModel):
    prompt: str
    negative_prompt: Optional[str]
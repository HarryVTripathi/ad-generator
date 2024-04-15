from .pipe import SDPipe


def generate_image(generation_id, prompt, negative_prompt):
    pipe = SDPipe()
    response = pipe.generate(
        generation_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        clip_skip=1,
        num_images_per_prompt=1,
        num_inference_steps=12,
    )
    return response

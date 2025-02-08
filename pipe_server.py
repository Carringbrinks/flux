from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import torch
from diffusers import FluxPipeline
from io import BytesIO
from fastapi.responses import StreamingResponse
import base64

# 初始化 FastAPI 应用
app = FastAPI()

# 加载模型
pipe = FluxPipeline.from_pretrained("/home/scb123/huggingface_weight/FLUX.1-schnell", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

# 定义输入数据模型
class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt to generate the image.")
    prompt_2: str = Field("", description="The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is will be used instead")
    num_images_per_prompt: int = Field(1, description="The number of images to generate per prompt.")
    guidance_scale: float = Field(7.0, description="The strength of the guidance scale.")
    num_inference_steps: int = Field(20, description="The number of inference steps for image generation.")
    max_sequence_length: Optional[int] = Field(256, description="The maximum sequence length.")
    output_type: Optional[str] = Field("pil", description="  The output format of the generate image. Choose between `PIL.Image.Image` or `np.array`.")
    seed: Optional[int] = Field(0, description="Random seed for reproducibility.")

@app.post("/flux.1-schnell/generate-image/")
async def generate_image(request: ImageGenerationRequest):
    try:
        # 使用传入参数生成图片
        generator = torch.Generator("cpu").manual_seed(request.seed)
        images = pipe(
            prompt=request.prompt,
            prompt_2 = request.prompt_2,
            num_images_per_prompt = request.num_images_per_prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            max_sequence_length=request.max_sequence_length,
            output_type = request.output_type,
            generator=generator,
        ).images

        base64_images = []
        for image in images:
            # 将图片保存到内存并返回
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes.seek(0)
            base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            base64_images.append(base64_image)
        return {"images": base64_images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
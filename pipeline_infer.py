# export LD_LIBRARY_PATH=/home/scb123/miniconda3/envs/text-to-image-flux/lib
# import torch
# from diffusers import FluxPipeline
import requests
import base64


# pipe = FluxPipeline.from_pretrained("/home/scb123/huggingface_weight/FLUX.1-schnell", torch_dtype=torch.float16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# # prompt = "A cat holding a sign that says hello world"
# prompt = "Young woman wearing black stockings and red high heels lying on the bed in the hotel"
# image = pipe(
#     prompt,
#     guidance_scale=7.0,
#     num_inference_steps=20,
#     max_sequence_length=256,
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]
# image.save("flux-schnell.png")


# 定义请求参数
url = "http://127.0.0.1:8000/flux.1-schnell/generate-image/"
payload = {
    "prompt": "A beautiful woman, wearing black stockings on her lower body and bare on top, lies on the hotel sofa.",
    "num_images_per_prompt": 2,
    "guidance_scale": 10.0,
    "num_inference_steps": 8,
    "max_sequence_length": 256,
    "seed": 0
}

# 发送 POST 请求
response = requests.post(url, json=payload)

if response.status_code == 200:
    # 提取 Base64 图片数据
    images_base64 = response.json()["images"]

    for i, image_bas64 in enumerate(images_base64):
        # 将 Base64 解码并保存为本地文件
        with open(f"{i}.png", "wb") as f:
            f.write(base64.b64decode(image_bas64))
            print(f"Image saved as {i}.png")
else:
    print(f"Request failed with status code: {response.status_code}, details: {response.text}")

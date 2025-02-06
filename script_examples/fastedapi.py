from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import websocket
import uuid
import json
import urllib.request
import urllib.parse
import os
import shutil
from pathlib import Path
from PIL import Image
import io


app = FastAPI()

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def queue_prompt(prompt):
    """ส่งคำขอไปยัง ComfyUI เพื่อเริ่มสร้างภาพ"""
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_history(prompt_id):
    """ดึงประวัติของการสร้างภาพ"""
    with urllib.request.urlopen(
        f"http://{server_address}/history/{prompt_id}"
    ) as response:
        return json.loads(response.read())


def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]
    output_images = {}
    current_node = ""
    while True:
        print(current_node)
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["prompt_id"] == prompt_id:
                    if data["node"] is None:
                        break  # Execution is done
                    else:
                        current_node = data["node"]
        else:
            if current_node == "21":
                print("abc")
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                # print(images_output)
                output_images[current_node] = images_output
                # print(output_images)
    return output_images


@app.head("/")
@app.get("/")
async def root():
    return {"message": "ComfyUI FastAPI Server is Running"}


@app.post("/gen")
async def genIng(img: UploadFile = File(...), seed: str = Form(...)):
    ws = websocket.WebSocket()
    print(img)
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    file_location = UPLOAD_DIR / img.filename
    with file_location.open("wb") as buffer:
        shutil.copyfileobj(img.file, buffer)

    prompt = {
        "3": {
            "inputs": {
                "seed": 444111,
                "steps": 40,
                "cfg": 7,
                "sampler_name": "dpmpp_2m",
                "scheduler": "normal",
                "denoise": 0.8700000000000001,
                "model": ["19", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["16", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        "6": {
            "inputs": {
                "text": "a human with **(detailed and highly stylized cat ears)**, **(vivid and high-contrast cat-like nose)**.The overall color palette follows a **bold Pop Art style**, using **highly saturated red, yellow, and blue tones**, with **strong black outlines and high contrast shading**.  ",
                "clip": ["17", 1],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "7": {
            "inputs": {
                "text": "multiple people, extra face, extra head, double face, crowd, group of people, other faces, split face, distorted face, duplicate face, out of frame, blurry, low quality, watermark, text, logo, unnatural, deformed body, extra limbs, extra arms, extra legs, multiple heads,  knife, gun",
                "clip": ["17", 1],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "8": {
            "inputs": {"samples": ["3", 0], "vae": ["14", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "9": {
            "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
        "10": {
            "inputs": {
                "image": "nananan.png",
                "upload": "image",
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "14": {
            "inputs": {"ckpt_name": "dragonfruitUnisex_dragonfruitgtV10.safetensors"},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"},
        },
        "16": {
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"},
        },
        "17": {
            "inputs": {
                "lora_name": "pop art style_v2.0.safetensors",
                "strength_model": 0.5,
                "strength_clip": 0.5,
                "model": ["14", 0],
                "clip": ["14", 1],
            },
            "class_type": "LoraLoader",
            "_meta": {"title": "Load LoRA"},
        },
        "18": {
            "inputs": {"preset": "STANDARD (medium strength)", "model": ["17", 0]},
            "class_type": "IPAdapterUnifiedLoader",
            "_meta": {"title": "IPAdapter Unified Loader"},
        },
        "19": {
            "inputs": {
                "weight": 1,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0,
                "end_at": 1,
                "embeds_scaling": "V only",
                "model": ["18", 0],
                "ipadapter": ["18", 1],
                "image": ["10", 0],
            },
            "class_type": "IPAdapterAdvanced",
            "_meta": {"title": "IPAdapter Advanced"},
        },
        "21": {
            "inputs": {"images": ["8", 0]},
            "class_type": "SaveImageWebsocket",
            "_meta": {"title": "SaveImageWebsocket"},
        },
    }

    images = get_images(ws, prompt)
    ws.close()
    imageName = "generated_image.png"
    for node_id in images:
        for image_data in images[node_id]:
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image_data))
            image.save(
                f"C:\\Users\\admin\\My project\\Assets\\imageOutput\\{imageName}"
            )

    generated_image_url = (
        f"http://localhost:8000/imageOutput/{imageName}"  # ใช้ URL ที่เซิร์ฟเวอร์ของคุณจัดให้
    )
    return {"image_url": generated_image_url}


async def generate_image():
    print("prompt")
    """API สำหรับสั่งให้ ComfyUI สร้างภาพ"""

    response = queue_prompt(prompt)
    prompt_id = response["prompt_id"]
    # history = get_history(prompt_id)
    images = []

    # if prompt_id in history["history"]:
    #     for img_data in history["history"][prompt_id]["outputs"]["9"]:
    #         image_bytes = get_image(
    #             img_data["filename"], img_data["subfolder"], img_data["type"]
    #         )
    #         image = Image.open(io.BytesIO(image_bytes))
    #         image_filename = f"generated_image_{img_data['filename']}"
    #         image.save(image_filename)
    #         images.append(image_filename)


@app.get("/get-image/{filename}")
async def get_saved_image(filename: str):
    """API สำหรับดึงภาพที่สร้างขึ้น"""
    image_path = UPLOAD_DIR / filename

    if image_path.exists():
        return FileResponse(image_path)
    else:
        return JSONResponse(content={"error": "Image not found"}, status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket สำหรับรับคำขอจาก Client"""
    await websocket.accept()
    try:
        prompt_id = queue_prompt({"example": "prompt"})["prompt_id"]
        history = get_history(prompt_id)
        images = []

        if prompt_id in history["history"]:
            for img_data in history["history"][prompt_id]["outputs"]["9"]:
                image_bytes = get_image(
                    img_data["filename"], img_data["subfolder"], img_data["type"]
                )
                image = Image.open(io.BytesIO(image_bytes))
                image_filename = f"generated_ws_image_{img_data['filename']}"
                image.save(image_filename)
                images.append(image_filename)

        await websocket.send_json(
            {"message": "Image generated via WebSocket", "images": images}
        )
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_text(f"An error occurred: {str(e)}")
    finally:
        await websocket.close()

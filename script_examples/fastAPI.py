from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websocket
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io

app = FastAPI()

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())


def queue_prompt(prompt):
    """ส่งคำขอไปยัง ComfyUI เพื่อเริ่มสร้างภาพ"""
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    """ดึงรูปภาพที่สร้างขึ้นจาก ComfyUI"""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        f"http://{server_address}/view?{url_values}"
    ) as response:
        return response.read()


def get_history(prompt_id):
    """ดึงประวัติของการสร้างภาพ"""
    with urllib.request.urlopen(
        f"http://{server_address}/history/{prompt_id}"
    ) as response:
        return json.loads(response.read())


def get_images(ws, prompt):
    """จัดการ WebSocket เพื่อรับข้อมูลภาพที่ถูกสร้าง"""
    prompt_id = queue_prompt(prompt)["prompt_id"]
    output_images = {}
    current_node = ""

    while True:
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
            if current_node == "9":  # ตรวจสอบว่าข้อมูลเป็นภาพ
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[current_node] = images_output

    return output_images


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint ที่ใช้รับคำขอจาก Client"""
    await websocket.accept()

    prompt_text = """
    {
        "3": {
            "inputs": {
                "seed": 606521811840601,
                "steps": 40,
                "cfg": 7,
                "sampler_name": "dpmpp_2m",
                "scheduler": "normal",
                "denoise": 0.87,
                "model": ["19", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["16", 0]
            },
            "class_type": "KSampler"
        },
        "6": {
            "inputs": {
                "text": "a human with **(detailed and highly stylized cat ears)**, **(vivid and high-contrast cat-like nose)**. The overall color palette follows a **bold Pop Art style**, using **highly saturated red, yellow, and blue tones**, with **strong black outlines and high contrast shading**.",
                "clip": ["17", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": "multiple people, extra face, extra head, double face, crowd, group of people, other faces, distorted face, low quality, watermark, text, unnatural",
                "clip": ["17", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["14", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        },
        "14": {
            "inputs": {
                "ckpt_name": "dragonfruitUnisex_dragonfruitgtV10.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "16": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "17": {
            "inputs": {
                "lora_name": "pop art style_v2.0.safetensors",
                "strength_model": 0.5,
                "strength_clip": 0.5,
                "model": ["14", 0],
                "clip": ["14", 1]
            },
            "class_type": "LoraLoader"
        },
        "18": {
            "inputs": {
                "preset": "STANDARD (medium strength)",
                "model": ["17", 0]
            },
            "class_type": "IPAdapterUnifiedLoader"
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
                "ipadapter": ["18", 1]
            },
            "class_type": "IPAdapterAdvanced"
        }
    }
    """

    prompt = json.loads(prompt_text)

    try:
        # ตั้งค่า WebSocket Client
        ws = websocket.WebSocket()
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}")

        # รับรูปภาพจาก WebSocket
        images = get_images(ws, prompt)
        ws.close()

        # แสดงและบันทึกภาพ
        for node_id in images:
            for image_data in images[node_id]:
                image = Image.open(io.BytesIO(image_data))
                image.save(f"generated_image_{node_id}.png")
                image.show()

        await websocket.send_text("Image processed and saved successfully.")

    except Exception as e:
        await websocket.send_text(f"An error occurred: {str(e)}")

    finally:
        await websocket.close()

# This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
# them being saved to disk

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())


def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        "http://{}/view?{}".format(server_address, url_values)
    ) as response:
        return response.read()


def get_history(prompt_id):
    with urllib.request.urlopen(
        "http://{}/history/{}".format(server_address, prompt_id)
    ) as response:
        return json.loads(response.read())


def get_images(ws, prompt):
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
            if current_node == 9:
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                print(images_output)
                output_images[current_node] = images_output
                print(output_images)
    return output_images


prompt_text = """
{
  "3": {
    "inputs": {
      "seed": 606521811840601,
      "steps": 40,
      "cfg": 7,
      "sampler_name": "dpmpp_2m",
      "scheduler": "normal",
      "denoise": 0.8700000000000001,
      "model": [
        "19",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "16",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "6": {
    "inputs": {
      "text": "a human with **(detailed and highly stylized cat ears)**, **(vivid and high-contrast cat-like nose)**.The overall color palette follows a **bold Pop Art style**, using **highly saturated red, yellow, and blue tones**, with **strong black outlines and high contrast shading**.  ",
      "clip": [
        "17",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "text": "multiple people, extra face, extra head, double face, crowd, group of people, other faces, split face, distorted face, duplicate face, out of frame, blurry, low quality, watermark, text, logo, unnatural, deformed body, extra limbs, extra arms, extra legs, multiple heads,  knife, gun",
      "clip": [
        "17",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "14",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "10": {
    "inputs": {
      "image": "Outdoors-man-portrait_(cropped).jpg",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "14": {
    "inputs": {
      "ckpt_name": "dragonfruitUnisex_dragonfruitgtV10.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "16": {
    "inputs": {
      "width": 512,
      "height": 512,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "17": {
    "inputs": {
      "lora_name": "pop art style_v2.0.safetensors",
      "strength_model": 0.5,
      "strength_clip": 0.5,
      "model": [
        "14",
        0
      ],
      "clip": [
        "14",
        1
      ]
    },
    "class_type": "LoraLoader",
    "_meta": {
      "title": "Load LoRA"
    }
  },
  "18": {
    "inputs": {
      "preset": "STANDARD (medium strength)",
      "model": [
        "17",
        0
      ]
    },
    "class_type": "IPAdapterUnifiedLoader",
    "_meta": {
      "title": "IPAdapter Unified Loader"
    }
  },
  "19": {
    "inputs": {
      "weight": 1,
      "weight_type": "linear",
      "combine_embeds": "concat",
      "start_at": 0,
      "end_at": 1,
      "embeds_scaling": "V only",
      "model": [
        "18",
        0
      ],
      "ipadapter": [
        "18",
        1
      ],
      "image": [
        "10",
        0
      ]
    },
    "class_type": "IPAdapterAdvanced",
    "_meta": {
      "title": "IPAdapter Advanced"
    }
  },
  "21": {
    "inputs": {
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImageWebsocket",
    "_meta": {
      "title": "SaveImageWebsocket"
    }
  }
}
"""

prompt = json.loads(prompt_text)
# set the text prompt for our positive CLIPTextEncode
# prompt["6"]["inputs"]["text"] = "masterpiece best quality man"

# set the seed for our KSampler node
# prompt["3"]["inputs"]["seed"] = 5

ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
images = get_images(ws, prompt)
ws.close()  # for in case this example is used in an environment where it will be repeatedly called, like in a Gradio app. otherwise, you'll randomly receive connection timeouts
# Commented out code to display the output images:

for node_id in images:
    for image_data in images[node_id]:
        from PIL import Image  # type: ignore
        import io

        image = Image.open(io.BytesIO(image_data))
        image.save(
            "C:\\Users\\admin\\OneDrive\\เดสก์ท็อป\\Work_p2_2024\\python_AI\\inClass\\LAB_01\\faceswap\\imageFromComfy\\comfy_2.png"
        )
        image.show()

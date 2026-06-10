#!/usr/bin/env python3
"""HunyuanVideo-1.5 I2V: image to video for jumping heroine."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/kang/Documents/ComfyUI/output")
SOURCE = "/home/kang/Documents/novel-to-video/outputs/fullbody-heroine/20260607-203308-01.png"

# Model paths
UNET_NAME = "hunyuan-video-1.5/transformer/480p_i2v_step_distilled/diffusion_pytorch_model.safetensors"
VAE_NAME = "hunyuan-video-1.5/vae/diffusion_pytorch_model.safetensors"
CLIP1_NAME = "hunyuan-video-1.5/text_encoder/model-00001-of-00004.safetensors"
CLIP2_NAME = "hunyuan-video-1.5/text_encoder_2/model.safetensors"
# Fallback to FP8 single-file model from Kijai if available
UNET_FP8 = "hunyuan_video_I2V_fp8_e4m3fn.safetensors"
VAE_FP8 = "hunyuan_video_vae_bf16.safetensors"

WIDTH, HEIGHT = 848, 480
LENGTH = 33  # frames
BATCH = 1

JUMP_PROMPT = (
    "A beautiful young woman in white blouse and black skirt, full body shot, "
    "she jumps high into the air with arms raised, dynamic midair action, "
    "solid light gray background, cinematic lighting, high quality 4K"
)
NEGATIVE_PROMPT = (
    "low quality, blurry, bad anatomy, extra limbs, missing limbs, deformed face, "
    "ugly, child, watermark, text, wrong clothes, broken clothes, static, still frame"
)


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"ComfyUI rejected: HTTP {exc.code}\n{body}") from exc


def get_json(path):
    with urllib.request.urlopen(SERVER + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_for_prompt(prompt_id, timeout_seconds=3600):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(3)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def use_fp8():
    """Check if FP8 model is available as fallback."""
    p = Path("/home/kang/Documents/ai-video-local/models") / UNET_FP8
    return p.exists()


def main():
    use_fp8_flag = use_fp8()
    if use_fp8_flag:
        print("Using Kijai FP8 model")
        unet = UNET_FP8
        vae = VAE_FP8
        # For FP8, we still need text encoders from hunyuan-video-1.5
    else:
        unet = UNET_NAME
        vae = VAE_NAME

    prompt_wf = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": SOURCE},
        },
        "2": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet,
                "weight_dtype": "fp8_e4m3fn" if use_fp8_flag else "default",
            },
        },
        "3": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": CLIP1_NAME,
                "clip_name2": CLIP2_NAME,
                "type": "hunyuan_video",
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
        },
        # Convert image to latent_space reference
        "5": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["1", 0], "vae": ["4", 0]},
        },
        # Encode text
        "6": {
            "class_type": "TextEncodeHunyuanVideo_ImageToVideo",
            "inputs": {
                "clip": ["3", 0],
                "clip_vision_output": None,
                "prompt": JUMP_PROMPT,
                "image_interleave": 4,
            },
        },
        "7": {
            "class_type": "TextEncodeHunyuanVideo_ImageToVideo",
            "inputs": {
                "clip": ["3", 0],
                "clip_vision_output": None,
                "prompt": NEGATIVE_PROMPT,
                "image_interleave": 4,
            },
        },
        # Hidden: use ModelSamplingDiscrete or scheduler from the model
        "8": {
            "class_type": "HunyuanVideo15ImageToVideo",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["7", 0],
                "vae": ["4", 0],
                "width": WIDTH,
                "height": HEIGHT,
                "length": LENGTH,
                "batch_size": BATCH,
                "start_image": ["1", 0],
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 2], "vae": ["4", 0]},
        },
        "10": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["9", 0],
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": "local_manhua/hunyuan-jump",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    }

    result = post_json("/prompt", {"prompt": prompt_wf})
    prompt_id = result["prompt_id"]
    print(f"queued prompt_id={prompt_id}")
    history = wait_for_prompt(prompt_id)

    outputs = []
    for output in history.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for item in output.get(key, []):
                f = item.get("filename")
                if f:
                    outputs.append(OUTPUT_DIR / item.get("subfolder", "") / f)

    if not outputs:
        print(json.dumps(history.get("outputs", {}), ensure_ascii=False, indent=2))
        raise SystemExit("No output files found")

    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()

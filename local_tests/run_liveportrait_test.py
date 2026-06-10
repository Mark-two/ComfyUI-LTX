#!/usr/bin/env python3
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/kang/Documents/ComfyUI/output")
SOURCE_IMAGE = "/home/kang/Documents/ComfyUI/output/local_manhua/majicmix_keyframe_00001_.png"
DRIVING_VIDEO = "/home/kang/Documents/ComfyUI/output/local_manhua/liveportrait-driver-20260607-181200.mp4"


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"ComfyUI rejected prompt: HTTP {exc.code}\n{body}") from exc


def get_json(path):
    with urllib.request.urlopen(SERVER + path, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_prompt(prompt_id, timeout_seconds=1800):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def main():
    driving_video = sys.argv[1] if len(sys.argv) > 1 else DRIVING_VIDEO
    delta_multiplier = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
    eye_multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
    lip_multiplier = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
    source_image = sys.argv[5] if len(sys.argv) > 5 else SOURCE_IMAGE
    prefix_suffix = Path(driving_video).stem.replace(".", "_").replace("-", "_")[:40]
    source_suffix = Path(source_image).stem.replace(".", "_").replace("-", "_")[:24]
    prefix_suffix = f"{prefix_suffix}-d{delta_multiplier:g}-e{eye_multiplier:g}-l{lip_multiplier:g}"
    prompt = {
        "1": {
            "class_type": "VHS_LoadImagePath",
            "inputs": {"image": source_image, "custom_width": 0, "custom_height": 0},
        },
        "2": {
            "class_type": "VHS_LoadVideoPath",
            "inputs": {
                "video": driving_video,
                "force_rate": 16,
                "custom_width": 512,
                "custom_height": 512,
                "frame_load_cap": 64,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "3": {
            "class_type": "DownloadAndLoadLivePortraitModels",
            "inputs": {"precision": "auto", "mode": "human"},
        },
        "4": {
            "class_type": "LivePortraitLoadFaceAlignmentCropper",
            "inputs": {
                "face_detector": "blazeface_back_camera",
                "landmarkrunner_device": "torch_gpu",
                "face_detector_device": "cuda",
                "face_detector_dtype": "fp16",
                "keep_model_loaded": True,
            },
        },
        "5": {
            "class_type": "LivePortraitCropper",
            "inputs": {
                "pipeline": ["3", 0],
                "cropper": ["4", 0],
                "source_image": ["1", 0],
                "dsize": 512,
                "scale": 2.1,
                "vx_ratio": 0.0,
                "vy_ratio": -0.08,
                "face_index": 0,
                "face_index_order": "large-small",
                "rotate": True,
            },
        },
        "6": {
            "class_type": "LivePortraitCropper",
            "inputs": {
                "pipeline": ["3", 0],
                "cropper": ["4", 0],
                "source_image": ["2", 0],
                "dsize": 512,
                "scale": 2.1,
                "vx_ratio": 0.0,
                "vy_ratio": -0.08,
                "face_index": 0,
                "face_index_order": "large-small",
                "rotate": True,
            },
        },
        "7": {
            "class_type": "LivePortraitRetargeting",
            "inputs": {
                "driving_crop_info": ["6", 1],
                "eye_retargeting": True,
                "eyes_retargeting_multiplier": eye_multiplier,
                "lip_retargeting": True,
                "lip_retargeting_multiplier": lip_multiplier,
            },
        },
        "8": {
            "class_type": "LivePortraitProcess",
            "inputs": {
                "pipeline": ["3", 0],
                "crop_info": ["5", 1],
                "source_image": ["1", 0],
                "driving_images": ["2", 0],
                "lip_zero": False,
                "lip_zero_threshold": 0.03,
                "stitching": True,
                "delta_multiplier": delta_multiplier,
                "mismatch_method": "mirror",
                "relative_motion_mode": "relative",
                "driving_smooth_observation_variance": 0.000003,
                "opt_retargeting_info": ["7", 0],
                "expression_friendly": True,
                "expression_friendly_multiplier": 0.8,
            },
        },
        "9": {
            "class_type": "LivePortraitComposite",
            "inputs": {
                "source_image": ["1", 0],
                "cropped_image": ["8", 0],
                "liveportrait_out": ["8", 1],
            },
        },
        "10": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["9", 0],
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": f"local_manhua/liveportrait-test-{source_suffix}-{prefix_suffix}",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
        "11": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": f"local_manhua/liveportrait-crop-{source_suffix}-{prefix_suffix}",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
        "12": {
            "class_type": "KeypointsToImage",
            "inputs": {"crop_info": ["6", 1], "draw_lines": True},
        },
        "13": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["12", 0],
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": f"local_manhua/liveportrait-driving-kps-{source_suffix}-{prefix_suffix}",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    }

    try:
        result = post_json("/prompt", {"prompt": prompt})
    except urllib.error.URLError as exc:
        raise SystemExit(f"ComfyUI is not reachable at {SERVER}: {exc}") from exc

    prompt_id = result["prompt_id"]
    print(f"queued prompt_id={prompt_id}")
    history = wait_for_prompt(prompt_id)

    outputs = []
    for output in history.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for item in output.get(key, []):
                filename = item.get("filename")
                if filename:
                    outputs.append(OUTPUT_DIR / item.get("subfolder", "") / filename)

    if not outputs:
        print(json.dumps(history.get("outputs", {}), ensure_ascii=False, indent=2))
        raise SystemExit("No output files found in ComfyUI history")

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LTX-2.3 I2V — uses browser-generated video's embedded prompt as template.
"""
import argparse, json, os, random, subprocess, sys, time, uuid
import urllib.request, urllib.error

SERVER_URL = "http://127.0.0.1:8188"
REF_VIDEO  = os.path.join(os.path.dirname(__file__),
                          "output/ltx-2.3/i2v_00003-audio.mp4")

# Node IDs in the reference prompt
POS_PROMPT = "121"
NEG_PROMPT = "110"
SEED_S1    = "115"   # RandomNoise stage1
SEED_S2    = "114"   # RandomNoise stage2
OUTPUT_S1  = "150"   # VHS_VideoCombine stage1 (save_output=False)
OUTPUT_S2  = "209"   # VHS_VideoCombine stage2 (save_output=True)
FRAMES     = "112"   # PrimitiveInt
IMAGE_NODE = "180"   # LoadImage

def load_ref_prompt():
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", REF_VIDEO],
                       capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)
    return json.loads(data["format"]["tags"]["prompt"])

def upload_image(filepath):
    boundary = "----" + uuid.uuid4().hex
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").encode() + data
    body += f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{SERVER_URL}/upload/image", data=body,
          headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
          method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

def queue_prompt(api_prompt):
    body = json.dumps({"prompt": api_prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{SERVER_URL}/prompt", data=body,
          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        b = e.read().decode()
        print(f"\n❌ Error {e.code}:")
        try:
            d = json.loads(b)
            err = d.get("error", {})
            print(f"  {err.get('message', '')}")
            if err.get("details"):
                print(f"  {err['details']}")
            for ni, ne in (d.get("node_errors", {}) or {}).items():
                for er in (ne.get("errors", []) or [])[:2]:
                    print(f"  Node #{ni} ({ne.get('class_type','?')}): {er.get('message','?')}")
        except json.JSONDecodeError:
            print(f"  {b[:500]}")
        raise

def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 I2V (browser-identical)")
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("-p", "--prompt", type=str, default=None,
                        help="Positive prompt (omit to keep default)")
    parser.add_argument("-n", "--negative", type=str, default=None,
                        help="Negative prompt (omit to keep default)")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 = random)")
    parser.add_argument("--frames", type=int, default=None, help="Frames (default: 121)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output prefix")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    prompt = load_ref_prompt()
    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)

    # ── Upload image ──────────────────────────────────────────
    print(f"📤 Uploading: {args.image}")
    result = upload_image(args.image)
    uploaded = result.get("name", os.path.basename(args.image))
    print(f"  ✅ {uploaded}")

    # Set image filename in LoadImage node
    prompt[IMAGE_NODE]["inputs"]["image"] = uploaded

    # ── Override nodes ─────────────────────────────────────────
    if args.prompt:
        prompt[POS_PROMPT]["inputs"]["text"] = args.prompt
    if args.negative:
        prompt[NEG_PROMPT]["inputs"]["text"] = args.negative
    if args.frames:
        prompt[FRAMES]["inputs"]["value"] = args.frames

    prompt[SEED_S1]["inputs"]["noise_seed"] = seed
    prompt[SEED_S2]["inputs"]["noise_seed"] = seed

    prefix = args.output or f"ltx23_i2v_{seed}"
    prompt[OUTPUT_S1]["inputs"]["filename_prefix"] = prefix
    prompt[OUTPUT_S2]["inputs"]["filename_prefix"] = prefix

    # ── Print info ─────────────────────────────────────────────
    pos = prompt[POS_PROMPT]["inputs"]["text"]
    neg = prompt[NEG_PROMPT]["inputs"]["text"]
    print(f"\n🚀 LTX-2.3 I2V (browser-identical)")
    print(f"  Template: i2v_00003-audio.mp4 embedded prompt")
    print(f"  Image: {uploaded}")
    print(f"  Seed: {seed}  Frames: {prompt[FRAMES]['inputs']['value']}")
    print(f"  Positive: {pos[:80]}...")
    print(f"  Negative: {neg[:60]}...")
    print(f"  Output: {prefix}_*.mp4\n")

    # ── Queue ──────────────────────────────────────────────────
    print("📤 Queuing...")
    r = queue_prompt(prompt)
    pid = r.get("prompt_id")
    if not pid:
        print(f"❌ Failed: {r}")
        sys.exit(1)
    print(f"  prompt_id={pid}")

    # ── Monitor ────────────────────────────────────────────────
    last = ""
    while True:
        try:
            req = urllib.request.Request(f"{SERVER_URL}/history/{pid}")
            with urllib.request.urlopen(req) as resp:
                h = json.loads(resp.read())
        except urllib.error.HTTPError:
            h = None

        if h and pid in h:
            s = h[pid].get("status", {})
            if s.get("completed"):
                print("\n✅ Done!")
                for nid, outs in h[pid].get("outputs", {}).items():
                    for val in outs.values():
                        if isinstance(val, list):
                            for v in val:
                                fn = v.get("filename", "") if isinstance(v, dict) else (v if isinstance(v, str) else "")
                                if fn:
                                    fp = os.path.abspath(os.path.join("output", fn))
                                    if os.path.exists(fp):
                                        sz = os.path.getsize(fp) / 1e6
                                        print(f"  ✓ {fp} ({sz:.1f} MB)")
                break
            if s.get("error_messages"):
                print(f"\n❌ Error: {s['error_messages']}")
                break
        else:
            try:
                with urllib.request.urlopen(f"{SERVER_URL}/queue") as resp:
                    q = json.loads(resp.read())
                info = f"r={len(q.get('queue_running',[]))},p={len(q.get('queue_pending',[]))}"
                if info != last:
                    print(f"\n⏳ {info}", end="", flush=True)
                    last = info
                else:
                    print(".", end="", flush=True)
            except Exception:
                print(".", end="", flush=True)
        time.sleep(3)

if __name__ == "__main__":
    main()

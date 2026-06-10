#!/usr/bin/env python3
"""
LTX-2.3 T2V — uses browser-generated video's embedded prompt as template.
Guarantees pixel-identical quality to the browser version.
"""
import argparse, json, os, random, subprocess, sys, time, urllib.request, urllib.error, uuid

SERVER_URL = "http://127.0.0.1:8188"
REF_VIDEO  = os.path.join(os.path.dirname(__file__),
                          "output/ltx-2.3/t2v_00007-audio.mp4")

# ── Node IDs in the reference prompt ─────────────────────────────
POS_PROMPT = "121"   # CLIPTextEncode (should be the main prompt)
NEG_PROMPT = "110"   # CLIPTextEncode (negative)
SEED_S1    = "115"   # RandomNoise (stage 1)
SEED_S2    = "114"   # RandomNoise (stage 2)
OUTPUT_1   = "186"   # VHS_VideoCombine
OUTPUT_2   = "187"   # VHS_VideoCombine
FRAMES     = "112"   # INTConstant

def load_ref_prompt():
    """Load the working API prompt embedded in the browser-generated video."""
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", REF_VIDEO],
                       capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)
    return json.loads(data["format"]["tags"]["prompt"])

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
    parser = argparse.ArgumentParser(description="LTX-2.3 T2V (browser-identical)")
    parser.add_argument("-p", "--prompt", type=str, default=None,
                        help="Positive prompt (omit to keep browser default)")
    parser.add_argument("-n", "--negative", type=str, default=None,
                        help="Negative prompt (omit to keep default)")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 = random)")
    parser.add_argument("--frames", type=int, default=None, help="Frames (default: 121)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output prefix")
    parser.add_argument("--width", type=int, default=None,
                        help="Latent width (default from ref: 960)")
    parser.add_argument("--height", type=int, default=None,
                        help="Latent height (default from ref: 544)")
    args = parser.parse_args()

    prompt = load_ref_prompt()

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)

    # ── Override nodes ─────────────────────────────────────────
    if args.prompt:
        prompt[POS_PROMPT]["inputs"]["text"] = args.prompt
    if args.negative:
        prompt[NEG_PROMPT]["inputs"]["text"] = args.negative
    if args.frames:
        prompt[FRAMES]["inputs"]["value"] = args.frames

    # Override latent dimensions (EmptyImage → ImageScaleBy(0.5) → GetImageSize → EmptyLTXVLatentVideo)
    if args.width and args.height:
        prompt["111"]["inputs"]["width"] = args.width * 2
        prompt["111"]["inputs"]["height"] = args.height * 2
        print(f"  📐 Latent: {args.width}×{args.height} → 🆙 {args.width*2}×{args.height*2}")
    elif args.width:
        prompt["111"]["inputs"]["width"] = args.width * 2
    elif args.height:
        prompt["111"]["inputs"]["height"] = args.height * 2

    prompt[SEED_S1]["inputs"]["noise_seed"] = seed
    prompt[SEED_S2]["inputs"]["noise_seed"] = seed

    prefix = args.output or f"ltx23_t2v_{seed}"
    prompt[OUTPUT_1]["inputs"]["filename_prefix"] = prefix
    prompt[OUTPUT_2]["inputs"]["filename_prefix"] = prefix

    # ── Print info ─────────────────────────────────────────────
    pos = prompt[POS_PROMPT]["inputs"]["text"]
    neg = prompt[NEG_PROMPT]["inputs"]["text"]
    print(f"🚀 LTX-2.3 T2V (browser-identical pipeline)")
    print(f"  Template: t2v_00007-audio.mp4 embedded prompt")
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

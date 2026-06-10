#!/usr/bin/env python3
"""
LTX-2.3 I2V — image-to-video with two-stage sampling.
"""
import argparse, json, os, random, sys, time, uuid
import urllib.request, urllib.error

SERVER_URL = "http://127.0.0.1:8188"

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

def build_prompt(image_name, text, neg, seed, frames=121,
                 width=768, height=512, prefix="LTX23i2v",
                 steps=20, cfg_video=4.0):
    """Build I2V pipeline from scratch: image→LTXVImgToVideoInplace→2-stage sample→decode."""
    p = {}
    nid = 0
    def n(): nonlocal nid; nid+=1; return str(nid)

    # ── Shared ────────────────────────────────────────────────
    CK      = n()  # 1 CheckpointLoaderSimple
    TE      = n()  # 2 LTXAVTextEncoderLoader
    CP      = n()  # 3 CLIPTextEncode (positive)
    CN      = n()  # 4 CLIPTextEncode (negative)
    COND    = n()  # 5 LTXVConditioning
    LEN     = n()  # 6 INTConstant
    EMPIMG  = n()  # 7 EmptyImage (proxy for dims)
    SCALE   = n()  # 8 ImageScaleBy (0.5)
    GIS     = n()  # 9 GetImageSize
    RAW_VID = n()  # 10 EmptyLTXVLatentVideo
    V_VAE   = n()  # 11 VAELoader
    A_VAE   = n()  # 12 LTXVAudioVAELoader
    RAW_AUD = n()  # 13 LTXVEmptyLatentAudio

    # ── Image input ───────────────────────────────────────────
    LOAD_IMG = n()  # 14 LoadImage
    PREPROC  = n()  # 15 LTXVPreprocess
    IMG2VID  = n()  # 16 LTXVImgToVideoInplace (s1)
    CONCAT1  = n()  # 17 LTXVConcatAVLatent

    # ── Stage 1 ───────────────────────────────────────────────
    L_DIS   = n()  # 18 LoraLoaderModelOnly (distilled)
    L_DET   = n()  # 19 LoraLoaderModelOnly (detailer)
    S1_SEL  = n()  # 20 KSamplerSelect
    S1_SCH  = n()  # 21 LTXVScheduler
    S1_NOI  = n()  # 22 RandomNoise
    S1_GD   = n()  # 23 CFGGuider
    S1_SAMP = n()  # 24 SamplerCustomAdvanced

    # ── Split & upscale ───────────────────────────────────────
    SP1     = n()  # 25 LTXVSeparateAVLatent
    UP_LDR  = n()  # 26 LatentUpscaleModelLoader
    UPS     = n()  # 27 LTXVLatentUpsampler

    # ── Stage 2 ───────────────────────────────────────────────
    IMG2VID2= n()  # 28 LTXVImgToVideoInplace (s2)
    CONCAT2 = n()  # 29 LTXVConcatAVLatent
    CROP    = n()  # 30 LTXVCropGuides
    S2_GD   = n()  # 31 CFGGuider (cfg=1.0)
    S2_SEL  = n()  # 32 KSamplerSelect
    S2_SIG  = n()  # 33 ManualSigmas
    S2_NOI  = n()  # 34 RandomNoise
    S2_SAMP = n()  # 35 SamplerCustomAdvanced

    # ── Decode ────────────────────────────────────────────────
    SP2     = n()  # 36 LTXVSeparateAVLatent
    UPS2    = n()  # 37 LTXVLatentUpsampler
    UP_LDR2 = n()  # 38 LatentUpscaleModelLoader
    DEC_V   = n()  # 39 VAEDecodeTiled
    DEC_A   = n()  # 40 LTXVAudioVAEDecode
    VID_OUT = n()  # 41 VHS_VideoCombine

    # ═══════════════ Build ═══════════════════════════════════
    p[CK]   = {"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"LTX-2.3/ltx-2.3-22b-dev-fp8.safetensors"}}
    p[TE]   = {"class_type":"LTXAVTextEncoderLoader","inputs":{"text_encoder":"LTX-2/gemma_3_12B_it_fp8_scaled.safetensors","ckpt_name":"LTX-2.3/ltx-2.3-22b-dev-fp8.safetensors","device":"default"}}
    p[CP]   = {"class_type":"CLIPTextEncode","inputs":{"text":text,"clip":[TE,0]}}
    p[CN]   = {"class_type":"CLIPTextEncode","inputs":{"text":neg,"clip":[TE,0]}}
    p[LEN]  = {"class_type":"INTConstant","inputs":{"value":frames}}
    # Proxy image for dimensions
    p[EMPIMG]={"class_type":"EmptyImage","inputs":{"width":width*2,"height":height*2,"batch_size":1,"color":0}}
    p[SCALE] ={"class_type":"ImageScaleBy","inputs":{"upscale_method":"area","scale_by":0.5,"image":[EMPIMG,0]}}
    p[GIS]   ={"class_type":"GetImageSize","inputs":{"image":[SCALE,0]}}
    p[RAW_VID]={"class_type":"EmptyLTXVLatentVideo","inputs":{"width":[GIS,0],"height":[GIS,1],"length":[LEN,0],"batch_size":1}}
    p[V_VAE] ={"class_type":"VAELoader","inputs":{"vae_name":"LTX-2.3/LTX23_video_vae_bf16.safetensors"}}
    p[A_VAE] ={"class_type":"LTXVAudioVAELoader","inputs":{"ckpt_name":"LTX-2.3/ltx-2.3-22b-dev-fp8.safetensors"}}
    p[RAW_AUD]={"class_type":"LTXVEmptyLatentAudio","inputs":{"frames_number":[LEN,0],"frame_rate":24,"batch_size":1,"audio_vae":[A_VAE,0]}}
    p[COND]  ={"class_type":"LTXVConditioning","inputs":{"frame_rate":24.0,"positive":[CP,0],"negative":[CN,0]}}

    # ── Image input ───────────────────────────────────────────
    p[LOAD_IMG]={"class_type":"LoadImage","inputs":{"image":image_name}}
    p[PREPROC] ={"class_type":"LTXVPreprocess","inputs":{"image":[LOAD_IMG,0],"img_compression":33}}
    # Inject image into empty video latent
    p[IMG2VID] ={"class_type":"LTXVImgToVideoInplace","inputs":{"vae":[V_VAE,0],"image":[PREPROC,0],"latent":[RAW_VID,0],"strength":1.0,"bypass":False}}

    # ── Audio + video concat (stage 1) ────────────────────────
    p[CONCAT1]={"class_type":"LTXVConcatAVLatent","inputs":{"video_latent":[IMG2VID,0],"audio_latent":[RAW_AUD,0]}}

    # ── LoRA chain (stage 1) ──────────────────────────────────
    p[L_DIS]={"class_type":"LoraLoaderModelOnly","inputs":{"lora_name":"LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors","strength_model":0.6,"model":[CK,0]}}
    p[L_DET]={"class_type":"LoraLoaderModelOnly","inputs":{"lora_name":"LTX-2/ltx-2-19b-ic-lora-detailer.safetensors","strength_model":1.0,"model":[L_DIS,0]}}

    p[S1_SEL]={"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler_ancestral_cfg_pp"}}
    p[S1_SCH]={"class_type":"LTXVScheduler","inputs":{"steps":steps,"max_shift":2.05,"base_shift":0.95,"stretch":True,"terminal":0.1,"latent":[RAW_VID,0]}}
    p[S1_NOI]={"class_type":"RandomNoise","inputs":{"noise_seed":seed+1}}
    p[S1_GD] ={"class_type":"CFGGuider","inputs":{"model":[L_DET,0],"positive":[COND,0],"negative":[COND,1],"cfg":cfg_video}}
    p[S1_SAMP]={"class_type":"SamplerCustomAdvanced","inputs":{"noise":[S1_NOI,0],"guider":[S1_GD,0],"sampler":[S1_SEL,0],"sigmas":[S1_SCH,0],"latent_image":[CONCAT1,0]}}

    p[SP1]={"class_type":"LTXVSeparateAVLatent","inputs":{"av_latent":[S1_SAMP,0]}}

    # ── Upscale model for latent upsampler ────────────────────
    p[UP_LDR]={"class_type":"LatentUpscaleModelLoader","inputs":{"model_name":"ltx-2.3-spatial-upscaler-x2-1.1.safetensors"}}
    p[UPS]   ={"class_type":"LTXVLatentUpsampler","inputs":{"samples":[SP1,0],"upscale_model":[UP_LDR,0],"vae":[V_VAE,0]}}

    # ── Stage 2: inject image into upsampled latent ────────────
    p[IMG2VID2]={"class_type":"LTXVImgToVideoInplace","inputs":{"vae":[V_VAE,0],"image":[PREPROC,0],"latent":[UPS,0],"strength":1.0,"bypass":False}}
    p[CONCAT2] ={"class_type":"LTXVConcatAVLatent","inputs":{"video_latent":[IMG2VID2,0],"audio_latent":[SP1,1]}}
    p[CROP]    ={"class_type":"LTXVCropGuides","inputs":{"positive":[COND,0],"negative":[COND,1],"latent":[CONCAT2,0]}}

    p[S2_GD]  ={"class_type":"CFGGuider","inputs":{"model":[L_DET,0],"positive":[CROP,0],"negative":[CROP,1],"cfg":1.0}}
    p[S2_SEL] ={"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler_cfg_pp"}}
    p[S2_SIG] ={"class_type":"ManualSigmas","inputs":{"sigmas":"0.909375, 0.725, 0.421875, 0.0"}}
    p[S2_NOI] ={"class_type":"RandomNoise","inputs":{"noise_seed":seed}}
    p[S2_SAMP]={"class_type":"SamplerCustomAdvanced","inputs":{"noise":[S2_NOI,0],"guider":[S2_GD,0],"sampler":[S2_SEL,0],"sigmas":[S2_SIG,0],"latent_image":[CONCAT2,0]}}

    # ── Decode ────────────────────────────────────────────────
    p[SP2]    ={"class_type":"LTXVSeparateAVLatent","inputs":{"av_latent":[S2_SAMP,0]}}
    p[UP_LDR2]={"class_type":"LatentUpscaleModelLoader","inputs":{"model_name":"ltx-2.3-spatial-upscaler-x2-1.1.safetensors"}}
    p[UPS2]   ={"class_type":"LTXVLatentUpsampler","inputs":{"samples":[SP2,0],"upscale_model":[UP_LDR2,0],"vae":[V_VAE,0]}}
    p[DEC_V]  ={"class_type":"VAEDecodeTiled","inputs":{"samples":[UPS2,0],"vae":[V_VAE,0],"tile_size":512,"overlap":64,"temporal_size":64,"temporal_overlap":16}}
    p[DEC_A]  ={"class_type":"LTXVAudioVAEDecode","inputs":{"samples":[SP2,1],"audio_vae":[A_VAE,0]}}
    p[VID_OUT]={"class_type":"VHS_VideoCombine","inputs":{"frame_rate":24.0,"loop_count":0,"filename_prefix":prefix,"format":"video/h264-mp4","pix_fmt":"yuv420p","crf":19,"save_metadata":True,"trim_to_audio":False,"pingpong":False,"save_output":True,"images":[DEC_V,0],"audio":[DEC_A,0]}}

    return p

def queue_prompt(prompt):
    body = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{SERVER_URL}/prompt", data=body,
          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp: return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        b = e.read().decode()
        print(f"\n❌ Error {e.code}:")
        try:
            d = json.loads(b); err = d.get("error", {})
            print(f"  {err.get('message', '')}")
            if err.get("details"): print(f"  {err['details']}")
            for ni, ne in (d.get("node_errors", {}) or {}).items():
                for er in (ne.get("errors", []) or [])[:2]:
                    print(f"  Node #{ni} ({ne.get('class_type','?')}): {er.get('message','?')}")
        except: print(f"  {b[:500]}")
        raise

def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 I2V")
    parser.add_argument("-i","--image", required=True)
    parser.add_argument("-p","--prompt", required=True)
    parser.add_argument("-n","--negative", default="blurry, low quality, still frame, frames, watermark, overlay, titles, has blurbox, has subtitles")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("-o","--output", default=None)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}"); sys.exit(1)

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)
    prefix = args.output or f"ltx23_i2v_{seed}"

    # Upload
    print(f"📤 Uploading image: {args.image}")
    r = upload_image(args.image)
    img_name = r.get("name", os.path.basename(args.image))
    print(f"  ✅ {img_name}")

    # Build
    prompt = build_prompt(img_name, args.prompt, args.negative, seed,
                          args.frames, args.width, args.height, prefix,
                          args.steps, args.cfg)

    print(f"\n🚀 LTX-2.3 I2V")
    print(f"  Image: {img_name}")
    print(f"  Seed: {seed}  Frames: {args.frames}  Steps: {args.steps}  CFG: {args.cfg}")
    print(f"  Size: {args.width}×{args.height} → 🆙 {args.width*2}×{args.height*2}")
    print(f"  Prompt: {args.prompt[:80]}...")
    print(f"  Output: {prefix}_*.mp4\n")

    # Queue
    print("📤 Queuing...")
    r = queue_prompt(prompt)
    pid = r.get("prompt_id")
    if not pid: print(f"❌ {r}"); sys.exit(1)
    print(f"  prompt_id={pid}")

    # Monitor
    last = ""
    while True:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/history/{pid}") as resp:
                h = json.loads(resp.read())
        except urllib.error.HTTPError: h = None
        if h and pid in h:
            s = h[pid].get("status", {})
            if s.get("completed"):
                print("\n✅ Done!")
                for nid, outs in h[pid].get("outputs", {}).items():
                    for val in outs.values():
                        if isinstance(val, list):
                            for v in val:
                                fn = v.get("filename","") if isinstance(v,dict) else (v if isinstance(v,str) else "")
                                if fn:
                                    fp = os.path.abspath(os.path.join("output", fn))
                                    sz = os.path.getsize(fp)/1e6 if os.path.exists(fp) else 0
                                    print(f"  ✓ {fp} ({sz:.1f} MB)" if sz else f"  ? {fn}")
                break
            if s.get("error_messages"): print(f"\n❌ Error: {s['error_messages']}"); break
        else:
            try:
                with urllib.request.urlopen(f"{SERVER_URL}/queue") as resp:
                    q = json.loads(resp.read())
                info = f"r={len(q.get('queue_running',[]))},p={len(q.get('queue_pending',[]))}"
                if info != last: print(f"\n⏳ {info}", end="", flush=True); last=info
                else: print(".", end="", flush=True)
            except: print(".", end="", flush=True)
        time.sleep(3)

if __name__ == "__main__":
    main()

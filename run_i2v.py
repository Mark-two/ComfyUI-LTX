#!/usr/bin/env python3
"""
LTX-2.3 I2V — exact replica of blueprint's two-stage pipeline.
"""
import argparse, json, os, random, sys, time, uuid
import urllib.request, urllib.error

SERVER_URL = "http://127.0.0.1:8188"

def upload_image(filepath):
    boundary = "----" + uuid.uuid4().hex; filename = os.path.basename(filepath)
    with open(filepath,"rb") as f: data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").encode()+data+f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{SERVER_URL}/upload/image", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urllib.request.urlopen(req) as r: return json.loads(r.read())

def build_prompt(img_name, text, neg, seed, frames=97, width=None, height=None, prefix="LTX23i2v", steps=20):
    p = {}; nid = 0
    def n(): nonlocal nid; nid+=1; return str(nid)

    # Image input: LoadImage → ImageResizeKJv2(704x1280, crop) → LTXVPreprocess
    CK=n(); TE=n(); CP=n(); CN=n()
    LI=n(); RJ=n(); PR=n()       # LoadImage, ImageResizeKJv2, LTXVPreprocess
    EI=n(); IS=n(); GI=n(); LV=n() # EmptyImage, ImageScaleBy, GetImageSize, EmptyLTXVLatentVideo
    VA=n(); AU=n(); LEN=n(); RA=n() # VAE, AudioVAE, Length, RawAudio

    # Stage 1: Camera LoRA → CFGGuider(4) → Inplace → ConcatAV → Sample
    LC=n(); GD1=n(); IV1=n()
    CA1=n()
    SK=n(); SH=n(); NS1=n(); CN1=n()
    SM1=n()

    # Split + Upscale
    SP1=n(); UL=n(); UP=n()

    # Stage 2: Distilled → Detailer LoRA → CFGGuider(1) → Inplace → ConcatAV → Sample(ManualSig)
    LD=n(); LK=n(); GD2=n(); IV2=n()
    CA2=n()
    CG=n()  # LTXVCropGuides
    S2K=n(); MS=n(); NS2=n(); SM2=n()

    # Decode
    SP2=n(); DE=n(); AD=n(); VO=n()

    IMG_W = width or 704
    IMG_H = height or 1280
    # Latent after scale 0.5
    LAT_W = IMG_W // 2; LAT_H = IMG_H // 2
    # Final after 2x upscale
    OUT_W = LAT_W * 2; OUT_H = LAT_H * 2

    # ── Shared ────────────────────────────────────────────────
    p[CK] ={"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"LTX-2.3/ltx-2.3-22b-dev-fp8.safetensors"}}
    p[TE] ={"class_type":"LTXAVTextEncoderLoader","inputs":{"text_encoder":"LTX-2/gemma_3_12B_it_fp8_scaled.safetensors","ckpt_name":"LTX-2.3/ltx-2.3-22b-dev-fp8.safetensors","device":"default"}}
    p[CP] ={"class_type":"CLIPTextEncode","inputs":{"text":text,"clip":[TE,0]}}
    p[CN] ={"class_type":"CLIPTextEncode","inputs":{"text":neg,"clip":[TE,0]}}

    # Image: load → resize to IMG_W×IMG_H (center crop) → preprocess
    p[LI] ={"class_type":"LoadImage","inputs":{"image":img_name}}
    p[RJ] ={"class_type":"ImageResizeKJv2","inputs":{"image":[LI,0],"width":IMG_W,"height":IMG_H,"upscale_method":"area","keep_proportion":"crop","pad_color":"0, 0, 0","crop_position":"center","divisible_by":2,"device":"cpu"}}
    p[PR] ={"class_type":"LTXVPreprocess","inputs":{"image":[RJ,0],"img_compression":33}}

    # Latent dimensions via proxy image (704×1280 → scale 0.5 → 352×640)
    p[EI] ={"class_type":"EmptyImage","inputs":{"width":IMG_W,"height":IMG_H,"batch_size":1,"color":0}}
    p[IS] ={"class_type":"ImageScaleBy","inputs":{"upscale_method":"area","scale_by":0.5,"image":[EI,0]}}
    p[GI] ={"class_type":"GetImageSize","inputs":{"image":[IS,0]}}
    p[LV] ={"class_type":"EmptyLTXVLatentVideo","inputs":{"width":[GI,0],"height":[GI,1],"length":frames,"batch_size":1}}

    # Audio
    p[LEN]={"class_type":"INTConstant","inputs":{"value":frames}}
    p[VA] ={"class_type":"VAELoader","inputs":{"vae_name":"LTX-2.3/LTX23_video_vae_bf16.safetensors"}}
    p[AU] ={"class_type":"LTXVAudioVAELoader","inputs":{"ckpt_name":"LTX-2.3/ltx-2.3-22b-dev-fp8.safetensors"}}
    p[RA] ={"class_type":"LTXVEmptyLatentAudio","inputs":{"frames_number":frames,"frame_rate":24,"batch_size":1,"audio_vae":[AU,0]}}

    # Conditioning
    p[CN1]= {"class_type":"LTXVConditioning","inputs":{"frame_rate":24.0,"positive":[CP,0],"negative":[CN,0]}}

    # ══════════ Stage 1: Coarse sampling (Camera LoRA, CFG=4.0) ══════════
    # Camera LoRA
    p[LC] ={"class_type":"LoraLoaderModelOnly","inputs":{"lora_name":"LTX-2/ltx-2-19b-lora-camera-control-dolly-in.safetensors","strength_model":1.0,"model":[CK,0]}}

    # Inject image into empty latent (uses checkpoint VAE slot 2)
    p[IV1]= {"class_type":"LTXVImgToVideoInplace","inputs":{"vae":[CK,2],"image":[PR,0],"latent":[LV,0],"strength":1.0,"bypass":False}}

    # Concat video + audio
    p[CA1]= {"class_type":"LTXVConcatAVLatent","inputs":{"video_latent":[IV1,0],"audio_latent":[RA,0]}}

    # Stage 1 CFGGuider
    p[GD1]= {"class_type":"CFGGuider","inputs":{"model":[LC,0],"positive":[CN1,0],"negative":[CN1,1],"cfg":4.0}}

    # Stage 1 sampler
    p[SK] ={"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler_ancestral_cfg_pp"}}
    p[SH] ={"class_type":"LTXVScheduler","inputs":{"steps":steps,"max_shift":2.05,"base_shift":0.95,"stretch":True,"terminal":0.1,"latent":[CA1,0]}}
    p[NS1]= {"class_type":"RandomNoise","inputs":{"noise_seed":seed+1}}
    p[SM1]= {"class_type":"SamplerCustomAdvanced","inputs":{"noise":[NS1,0],"guider":[GD1,0],"sampler":[SK,0],"sigmas":[SH,0],"latent_image":[CA1,0]}}

    # Split stage1 AV
    p[SP1]= {"class_type":"LTXVSeparateAVLatent","inputs":{"av_latent":[SM1,0]}}

    # ══════════ Upscale stage1 video latent ══════════
    p[UL] ={"class_type":"LatentUpscaleModelLoader","inputs":{"model_name":"ltx-2.3-spatial-upscaler-x2-1.1.safetensors"}}
    p[UP] ={"class_type":"LTXVLatentUpsampler","inputs":{"samples":[SP1,0],"upscale_model":[UL,0],"vae":[CK,2]}}

    # ══════════ Stage 2: Refinement (Distilled+Detailer LoRA, CFG=1.0, ManualSigmas) ══════════
    # Re-inject image into upsampled latent
    p[IV2]= {"class_type":"LTXVImgToVideoInplace","inputs":{"vae":[CK,2],"image":[PR,0],"latent":[UP,0],"strength":1.0,"bypass":False}}

    # Re-concat with stage1 audio
    p[CA2]= {"class_type":"LTXVConcatAVLatent","inputs":{"video_latent":[IV2,0],"audio_latent":[SP1,1]}}

    # Distilled LoRA → Detailer LoRA (chained)
    p[LD] ={"class_type":"LoraLoaderModelOnly","inputs":{"lora_name":"LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors","strength_model":0.6,"model":[CK,0]}}
    p[LK] ={"class_type":"LoraLoaderModelOnly","inputs":{"lora_name":"LTX-2/ltx-2-19b-ic-lora-detailer.safetensors","strength_model":1.0,"model":[LD,0]}}

    # CropGuides
    p[CG] ={"class_type":"LTXVCropGuides","inputs":{"positive":[CN1,0],"negative":[CN1,1],"latent":[CA2,0]}}

    # Stage 2 CFGGuider (CFG=1.0)
    p[GD2]= {"class_type":"CFGGuider","inputs":{"model":[LK,0],"positive":[CG,0],"negative":[CG,1],"cfg":1.0}}

    # Stage 2 sampler (ManualSigmas, 4 steps)
    p[S2K]= {"class_type":"KSamplerSelect","inputs":{"sampler_name":"euler_cfg_pp"}}
    p[MS] ={"class_type":"ManualSigmas","inputs":{"sigmas":"0.909375, 0.725, 0.421875, 0.0"}}
    p[NS2]= {"class_type":"RandomNoise","inputs":{"noise_seed":seed}}
    p[SM2]= {"class_type":"SamplerCustomAdvanced","inputs":{"noise":[NS2,0],"guider":[GD2,0],"sampler":[S2K,0],"sigmas":[MS,0],"latent_image":[CA2,0]}}

    # ══════════ Decode ══════════
    p[SP2]= {"class_type":"LTXVSeparateAVLatent","inputs":{"av_latent":[SM2,0]}}
    p[DE] ={"class_type":"VAEDecodeTiled","inputs":{"samples":[SP2,0],"vae":[CK,2],"tile_size":512,"overlap":64,"temporal_size":64,"temporal_overlap":16}}
    p[AD] ={"class_type":"LTXVAudioVAEDecode","inputs":{"samples":[SP2,1],"audio_vae":[AU,0]}}
    p[VO] ={"class_type":"VHS_VideoCombine","inputs":{"frame_rate":24.0,"loop_count":0,"filename_prefix":prefix,"format":"video/h264-mp4","pix_fmt":"yuv420p","crf":19,"save_metadata":True,"trim_to_audio":False,"pingpong":False,"save_output":True,"images":[DE,0],"audio":[AD,0]}}
    return p

def queue_prompt(prompt):
    body = json.dumps({"prompt":prompt,"client_id":str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{SERVER_URL}/prompt",data=body,headers={"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(req) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e:
        b=e.read().decode(); print(f"\n❌ Error {e.code}:")
        try:
            d=json.loads(b);err=d.get("error",{});print(f"  {err.get('message','')}")
            if err.get("details"):print(f"  {err['details']}")
            for ni,ne in (d.get("node_errors",{}) or {}).items():
                for er in (ne.get("errors",[]) or [])[:2]: print(f"  Node #{ni}({ne.get('class_type','?')}): {er.get('message','?')}")
        except: print(f"  {b[:500]}")
        raise

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i","--image",required=True); p.add_argument("-p","--prompt",required=True)
    p.add_argument("-n","--negative",default="blurry, low quality, still frame, frames, watermark, overlay, titles, has blurbox, has subtitles")
    p.add_argument("--seed",type=int,default=-1); p.add_argument("--frames",type=int,default=97)
    p.add_argument("--steps",type=int,default=20); p.add_argument("-o","--output",default=None)
    p.add_argument("--width",type=int,default=704); p.add_argument("--height",type=int,default=1280)
    args = p.parse_args()
    if not os.path.exists(args.image): print(f"❌ {args.image}"); sys.exit(1)

    seed = args.seed if args.seed>=0 else random.randint(0,2**31-1)
    prefix = args.output or f"ltx23_i2v_{seed}"

    print(f"📤 Uploading: {args.image}")
    r = upload_image(args.image)
    img_name = r.get("name", os.path.basename(args.image))

    prompt = build_prompt(img_name, args.prompt, args.negative, seed, args.frames, args.width, args.height, prefix, args.steps)
    print(f"\n🚀 LTX-2.3 I2V (2-stage blueprint)")
    print(f"  Image: {img_name}  Seed:{seed}  Frames:{args.frames}  Steps:{args.steps}")
    print(f"  Image→Resize({args.width}×{args.height})→Latent({args.width//2}×{args.height//2})→🆙{args.width}×{args.height}")
    print(f"  S1: CameraLoRA CFG=4.0 → S2: Distilled+Detailer CFG=1.0")

    print("\n📤 Queuing...")
    r = queue_prompt(prompt)
    pid = r.get("prompt_id")
    if not pid: print(f"❌ {r}"); sys.exit(1)
    print(f"  prompt_id={pid}")

    last = ""
    while True:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/history/{pid}") as resp:
                h = json.loads(resp.read())
        except: h = None
        if h and pid in h:
            s = h[pid].get("status", {})
            if s.get("completed"):
                print("\n✅ Done!")
                for nid,outs in h[pid].get("outputs",{}).items():
                    for val in outs.values():
                        if isinstance(val,list):
                            for v in val:
                                fn = v.get("filename","") if isinstance(v,dict) else (v if isinstance(v,str) else "")
                                if fn:
                                    fp = os.path.abspath(f"output/{fn}")
                                    sz = os.path.getsize(fp)/1e6 if os.path.exists(fp) else 0
                                    print(f"  ✓ {fp} ({sz:.1f} MB)" if sz else f"  ? {fn}")
                break
            if s.get("error_messages"): print(f"\n❌ {s['error_messages']}"); break
        else:
            try:
                with urllib.request.urlopen(f"{SERVER_URL}/queue") as resp:
                    q = json.loads(resp.read())
                info = f"r={len(q.get('queue_running',[]))},p={len(q.get('queue_pending',[]))}"
                if info != last: print(f"\n⏳ {info}", end="", flush=True); last=info
                else: print(".", end="", flush=True)
            except: print(".", end="", flush=True)
        time.sleep(3)

if __name__=="__main__":
    main()

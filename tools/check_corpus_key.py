"""Print the clip key and hash for a prompt/seed, without generating anything.

Used to prove the key is stable across machines and library versions, which is
the whole point of hashing it into the object path.

    python tools/check_corpus_key.py --prompt "..." --seed 0
"""
import argparse, sys
from tools.generate_corpus import ClipKey, MODEL, MODEL_REPO, MODEL_REVISION, WEIGHT_DTYPE, SAMPLER_STEPS, DISTILL_LORA, WIDTH, HEIGHT, FRAMES, FPS

p = argparse.ArgumentParser()
p.add_argument("--prompt", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--scene", default="coast")
p.add_argument("--prompt-id", default="p000")
a = p.parse_args()
k = ClipKey(model=MODEL, model_repo=MODEL_REPO, model_revision=MODEL_REVISION,
            weight_dtype=WEIGHT_DTYPE, sampler_steps=SAMPLER_STEPS, distill_lora=DISTILL_LORA,
            scene=a.scene, prompt_id=a.prompt_id, prompt=a.prompt, seed=a.seed,
            width=WIDTH, height=HEIGHT, frames=FRAMES, fps=FPS)
print("clip_id", k.clip_id)
print("sha256 ", k.sha256())
sys.exit(0)

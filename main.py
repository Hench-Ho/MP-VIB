"""
Inference entry point for MP-VIB (Multi-Prototype Variational Information
Bottleneck) SSL-based speech spoofing detection.

Usage (see README):
    $ python main.py \
        --wavlm_path       wavlm/WavLM-Large.pt \
        --model_path       checkpoints/mp_vib_best.pth \
        --protocol_path    /path/to/eval_protocol.txt \
        --database_path    /path/to/eval_flac_dir \
        --output_path      scores.txt
        
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader

from model import Model
from data_utils_SSL import load_dev_data, EvalDataset
from calculate_modules import compute_eer


def parse_args():
    p = argparse.ArgumentParser(description="MP-VIB inference")
    p.add_argument("--wavlm_path", type=str, default="wavlm/WavLM-Large.pt",
                   help="path to the pre-trained WavLM checkpoint (.pt)")
    p.add_argument("--model_path", type=str, default="checkpoints/mp_vib_best.pth",
                   help="path to the trained MP-VIB checkpoint (.pth)")
    p.add_argument("--protocol_path", type=str, required=True,
                   help="evaluation protocol file (key at col 2, label last)")
    p.add_argument("--database_path", type=str, required=True,
                   help="directory holding the eval .flac files")
    p.add_argument("--output_path", type=str, default="scores.txt",
                   help="where to write per-utterance scores")

    # architecture hyper-parameters (must match the trained checkpoint)
    p.add_argument("--z_dim", type=int, default=512,
                   help="latent bottleneck dimension (paper default: 512)")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--n_spoof_prototypes", type=int, default=6,
                   help="number of spoof prototypes K (paper default: 6)")
    p.add_argument("--n_heads", type=int, default=8)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(args):
    model = Model(
        wavlm_ckpt_path=args.wavlm_path,
        z_dim=args.z_dim,
        proj_dim=args.proj_dim,
        n_spoof_prototypes=args.n_spoof_prototypes,
        n_heads=args.n_heads,
    )

    if os.path.isfile(args.model_path):
        state = torch.load(args.model_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] missing keys when loading MP-VIB: {missing}")
        if unexpected:
            print(f"[warn] unexpected keys when loading MP-VIB: {unexpected}")
        print(f"Loaded MP-VIB checkpoint from {args.model_path}")
    else:
        print(f"[warn] MP-VIB checkpoint not found at {args.model_path}; "
              f"running with randomly initialised back-end.")

    return model.to(args.device).eval()


@torch.no_grad()
def run_inference(model, loader, device):
    utt_ids, scores, labels = [], [], []
    for x, meta in loader:
        x = x.to(device)
        s = model.score(x)                       # (B,) bonafide log-prob
        scores.extend(s.cpu().tolist())
        if torch.is_tensor(meta):
            labels.extend(meta.tolist())
        else:
            labels.extend(list(meta))
    return utt_ids, scores, labels


def main():
    args = parse_args()

    file_list, d_meta = load_dev_data(args.protocol_path, args.database_path)
    dataset = EvalDataset(file_list, d_meta)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args)

    print(f"Running inference over {len(dataset)} utterances "
          f"on {args.device} ...")
    _, scores, labels = run_inference(model, loader, args.device)

    with open(args.output_path, "w") as f:
        for utt, lab, sc in zip(file_list, labels, scores):
            tag = "bonafide" if lab == 1 else "spoof"
            f.write(f"{utt} {tag} {sc}\n")
    print(f"Scores written to {args.output_path}")

    import numpy as np
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    bona = scores[labels == 1]
    spoof = scores[labels == 0]
    if bona.size and spoof.size:
        eer, _, _, _ = compute_eer(bona, spoof)
        print(f"EER = {eer * 100:.2f}%")
    else:
        print("[warn] cannot compute EER: need both bonafide and spoof scores.")


if __name__ == "__main__":
    main()

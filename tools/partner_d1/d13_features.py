"""D1.3 step 1: pool features (20,565 images, row order = sorted(dataset.train)) of the epoch-50 model of each C0 seed, with train_al.py's
own clustering transform (get_test_loader: Resize 256x128 bicubic, ToTensor, Normalize, no flip), create_model and extract_features;
L2-normalised as train_al.py:340-347 does for the memory bank.  Output: $OUT/feat_s{seed}.npy (float32)."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import multiprocessing as mp

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2,3"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    import train_al                                            # the project's own functions
    from spcl.evaluators import extract_features
    from spcl.utils.serialization import load_checkpoint
    ds = train_al.get_data("wildlife", os.path.join(REPO, "dataroot"), "combined", 0, 102)
    train = sorted(ds.train); assert len(train) == 20565
    pool = np.load(f"{KIT_DATA}/pool.npz")
    assert [p for p, _, _ in train] == list(pool["paths"]) and [q for _, q, _ in train] == list(pool["pid"]), "pool order != kit pool.npz"
    for s in [int(x) for x in a.seeds.split(",")]:
        outp = f"{OUT}/feat_s{s}.npy"
        if os.path.exists(outp): log("exists", outp); continue
        args = argparse.Namespace(arch="resnet50", features=0, dropout=0)
        model = train_al.create_model(args)
        ck = load_checkpoint(f"{LOGS[s]}/checkpoint_50.pth.tar"); assert ck["epoch"] == 50
        model.load_state_dict(ck["state_dict"])
        loader = train_al.get_test_loader(ds, 256, 128, 64, a.workers, testset=train)
        feats, _ = extract_features(model, loader, print_freq=100)
        f = torch.cat([feats[fn].unsqueeze(0) for fn, _, _ in train], 0)
        f = F.normalize(f, dim=1).float().cpu().numpy()
        np.save(outp, f); log("seed", s, "saved", outp, f.shape, "ckpt epoch", ck["epoch"])
        del model, loader, feats

if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    main()

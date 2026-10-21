"""D1.5 control R_DINO: frozen DINOv2-large (facebook/dinov2-large, local HF cache, offline) CLS embedding of the 20,565 pool images.
Resize((224,224)) bilinear, ImageNet normalisation (same pre-processing family as the project's evaluation transform), no flip; L2-normalised. -> $OUT/dino_feat.npy"""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_HOME"] = "/storage/dd25007/cache/huggingface"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import multiprocessing as mp
from PIL import Image
import torchvision.transforms as TT

class DS(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.t = TT.Compose([TT.Resize((224, 224)), TT.ToTensor(), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.t(Image.open(self.paths[i]).convert("RGB")), i

def main():
    from transformers import AutoModel
    paths = list(np.load(f"{KIT_DATA}/pool.npz")["paths"])
    model = AutoModel.from_pretrained("facebook/dinov2-large", local_files_only=True).eval().cuda()
    dl = torch.utils.data.DataLoader(DS(paths), batch_size=64, num_workers=6, shuffle=False)
    out = np.zeros((len(paths), model.config.hidden_size), np.float32)
    with torch.no_grad():
        for b, (x, idx) in enumerate(dl):
            h = model(pixel_values=x.cuda()).pooler_output              # CLS token after the final layernorm
            out[idx.numpy()] = F.normalize(h.float(), dim=1).cpu().numpy()
            if b % 50 == 0: log("batch", b, "/", len(dl))
    np.save(f"{OUT}/dino_feat.npy", out); log("saved", out.shape)

if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True); main()

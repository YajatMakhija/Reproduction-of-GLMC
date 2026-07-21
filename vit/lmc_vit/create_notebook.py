import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Intro and parameters
cell_1 = """\
# 🚀 LMC ViT on Kaggle with TPU & Multi-Dataset Support
# 
# **Setup Instructions:**
# 1. Add your code as a dataset to Kaggle (e.g., `codefile` containing the `lmc_vit` folder).
# 2. Add your desired dataset to Kaggle (e.g., `cifar-100-python`, `tiny-imagenet`).
# 3. Ensure the accelerator is set to **TPU VM v3-8** in the notebook settings.

import os
import sys

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
DATASET = "CIFAR-100" # Options: "CIFAR-10", "CIFAR-100", "TINY-IMAGENET"

# If your dataset is mounted on Kaggle, provide the path here (e.g. /kaggle/input/cifar-100-python)
# If left as None, the script will attempt to download CIFAR datasets automatically.
KAGGLE_DATASET_INPUT_PATH = None 

# If you uploaded lmc_vit inside a Kaggle dataset called "codefile", it will be mounted here:
CODEBASE_KAGGLE_PATH = "/kaggle/input/codefile/lmc_vit"
"""

# Cell 2: Setup codebase and TPU dependencies
cell_2 = """\
# ==========================================
# 📦 SETUP & INSTALLATION
# ==========================================
import subprocess

# 1. Copy Codebase to working directory so we can modify and run it
if os.path.exists(CODEBASE_KAGGLE_PATH):
    print(f"Copying codebase from {CODEBASE_KAGGLE_PATH}...")
    !cp -r {CODEBASE_KAGGLE_PATH} /kaggle/working/lmc_vit
else:
    print("Codebase Kaggle input not found. Assuming 'lmc_vit' is already in the current directory.")

# 2. Add to Python path
if os.path.abspath("lmc_vit") not in sys.path:
    sys.path.append(os.path.abspath("lmc_vit"))
os.chdir("lmc_vit")

# 3. Install requirements
print("Installing dependencies...")
!pip install -q POT einops pyyaml tqdm

# 4. Install TPU dependencies (torch_xla is pre-installed on Kaggle TPU VMs, but we check anyway)
try:
    import torch_xla
    print("✅ PyTorch XLA (TPU Support) is ready!")
except ImportError:
    print("⚠️ PyTorch XLA not found. If you are on a TPU VM, make sure you selected TPU as the accelerator.")
"""

# Cell 3: Patch data.py
cell_3 = """\
# ==========================================
# 🛠️ PATCH DATA.PY (Add Tiny ImageNet & Kaggle Paths)
# ==========================================

patched_data_py = '''\\
\"\"\"CIFAR-10 / CIFAR-100 / Tiny ImageNet data loaders.\"\"\"

import os
import torch
import torchvision
import torchvision.transforms as T

STATS = {
    "CIFAR-10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR-100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "TINY-IMAGENET": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
NUM_CLASSES = {"CIFAR-10": 10, "CIFAR-100": 100, "TINY-IMAGENET": 200}
_DATASET = {"CIFAR-10": torchvision.datasets.CIFAR10, "CIFAR-100": torchvision.datasets.CIFAR100}

def get_loaders(dataset="CIFAR-10", batch_size=128, root="./data", num_workers=2,
                augment=True, download=True):
    dataset = dataset.upper()
    if dataset not in STATS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    mean, std = STATS[dataset]
    
    if dataset == "TINY-IMAGENET":
        train_tf = T.Compose([
            T.RandomCrop(64, padding=8) if augment else T.Resize(64),
            T.RandomHorizontalFlip() if augment else T.Lambda(lambda x: x),
            T.ToTensor(), T.Normalize(mean, std)
        ])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        
        train_path = os.path.join(root, "train")
        val_path = os.path.join(root, "val")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Tiny ImageNet train folder not found at {train_path}. Please provide a valid Kaggle Dataset path.")
            
        trainset = torchvision.datasets.ImageFolder(root=train_path, transform=train_tf)
        testset = torchvision.datasets.ImageFolder(root=val_path, transform=test_tf)
    else:
        cls = _DATASET[dataset]
        train_tf = T.Compose(([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        ] if augment else []) + [T.ToTensor(), T.Normalize(mean, std)])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

        trainset = cls(root=root, train=True, download=download, transform=train_tf)
        testset = cls(root=root, train=False, download=download, transform=test_tf)
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)
    return trainloader, testloader, NUM_CLASSES[dataset]
'''

with open("data.py", "w") as f:
    f.write(patched_data_py)
print("✅ Patched data.py for Tiny ImageNet and custom path support.")
"""

# Cell 4: Patch train_model.py and train_merger.py for TPU mark_step
cell_4 = """\
# ==========================================
# 🛠️ PATCH TRAINING LOOPS FOR TPU (xm.mark_step)
# ==========================================
# For PyTorch XLA, we must call xm.mark_step() after optimizer.step() 
# so the TPU executes the graph.

def patch_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    
    # Inject xm import
    if "import torch_xla" not in content:
        content = "try:\\n    import torch_xla.core.xla_model as xm\\n    HAS_XLA = True\\nexcept ImportError:\\n    HAS_XLA = False\\n" + content
    
    # Inject mark_step
    if "xm.mark_step()" not in content:
        content = content.replace(
            "optimizer.step()",
            "optimizer.step()\\n            if HAS_XLA and hasattr(x, 'device') and x.device.type == 'xla':\\n                xm.mark_step()"
        )
    
    with open(filepath, "w") as f:
        f.write(content)

patch_file("train_model.py")
patch_file("train_merger.py")
print("✅ Patched train_model.py and train_merger.py for PyTorch XLA (TPU) support.")
"""

# Cell 5: Setup TPU device and load config
cell_5 = """\
# ==========================================
# 🚀 INITIALIZE DEVICE & RUN PIPELINE
# ==========================================
import yaml
import torch
import train_model
from train_merger import train_merger
from eval_barrier import compute_barriers, report

# 1. Device Setup
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(f"🚀 Using TPU Device: {device}")
except ImportError:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using Device: {device}")

# 2. Update Configuration
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cfg["dataset"] = DATASET
print(f"Dataset set to: {cfg['dataset']}")

data_root = KAGGLE_DATASET_INPUT_PATH if KAGGLE_DATASET_INPUT_PATH else "./data"

# If using local data folder on Kaggle for CIFAR downloads
if data_root == "./data":
    os.makedirs(data_root, exist_ok=True)
    
# Monkey-patch get_loaders default root
import data
_original_get_loaders = data.get_loaders
data.get_loaders = lambda dataset, batch_size=128: _original_get_loaders(dataset=dataset, batch_size=batch_size, root=data_root)
"""

# Cell 6: Run execution
cell_6 = """\
# ==========================================
# 🏃 RUN PIPELINE
# ==========================================
out_models_dir = os.path.join("outputs", "models")
out_mergers_dir = os.path.join("outputs", "mergers")

print("========== 1/4: TRAINING MODEL 0 ==========")
f0 = train_model.train(cfg, seed=0, out_root=out_models_dir, device=device)

print("\\n========== 2/4: TRAINING MODEL 1 ==========")
f1 = train_model.train(cfg, seed=1, out_root=out_models_dir, device=device)

print("\\n========== 3/4: RUNNING LEARNED MATCHING ==========")
_, merger_dir = train_merger(
    f0, f1, 
    device=device, 
    epochs=cfg.get("merger_epochs", 15), 
    wm_iterations=15,
    batch_size=cfg.get("batch_size", 128),
    out_root=out_mergers_dir
)

print("\\n========== 4/4: BARRIER EVALUATION ==========")
results = compute_barriers(
    f0, f1, 
    device=device, 
    merger_dir=merger_dir,
    wm_iterations=15
)

print("\\n📊 FINAL RESULTS:")
report(results)
print("\\n✅ **Full pipeline execution complete!** All models and mergers have been saved to the `outputs/` folder.")
"""

nb.cells = [
    nbf.v4.new_code_cell(cell_1),
    nbf.v4.new_code_cell(cell_2),
    nbf.v4.new_code_cell(cell_3),
    nbf.v4.new_code_cell(cell_4),
    nbf.v4.new_code_cell(cell_5),
    nbf.v4.new_code_cell(cell_6)
]

with open("../LMC_VIT_Kaggle_TPU.ipynb", "w") as f:
    nbf.write(nb, f)
print("Created notebook at LMC_VIT_Kaggle_TPU.ipynb")

import torch
from torch import nn
import torch.nn.functional as F
import math

from pathlib import Path
from copy import deepcopy as dc

# custom imports
from models import *
# from train_utils import *

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

ROBOTS_NUM = 20
AREA_W = 30.0
MODELS_PATH = Path().resolve() / "models"
MODEL_SAVE_PATH = Path().resolve() / "SerializedModels"
COVERAGE_MODEL_PATH = MODELS_PATH / "pycov_model_old.pth"
PF_MODEL_PATH = MODELS_PATH / "pf-model-with-obs.pth"

coverage_model = DropoutCoverageModel(2*ROBOTS_NUM, 2*ROBOTS_NUM, device).to(device)
coverage_model.load_state_dict(torch.load(COVERAGE_MODEL_PATH, map_location=torch.device(device)))

# Load model
pf_model = PFModelWithObs(8,6,128).to(device)
pf_model.load_state_dict(torch.load(PF_MODEL_PATH, map_location=torch.device(device)))

# Convert to Torch Script via Annotation
cov_sm = torch.jit.script(coverage_model)
pf_sm = torch.jit.script(pf_model)

# Serializing model to a file
cov_sm.save(str(MODEL_SAVE_PATH/"coverage_model_old.pt"))
pf_sm.save(str(MODEL_SAVE_PATH / "pf_model_with_obs.pt"))
import os
import sys
import glob
import json
import logging
import subprocess
import shutil
import traceback
from datetime import datetime
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any, Optional

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from google.cloud import storage

import warnings
warnings.filterwarnings("ignore")

# --- HYDRA IMPORTS ---
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
# ---------------------

# --------- Diffusers (SDXL + ControlNet) ----------
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
    AutoencoderKL,
)

# --------- MeshGraphormer (matches demo_hand_inference.py) ----------
# Assuming MeshGraphormer is installed or PYTHONPATH is set correctly
try:
    import src.modeling.data.config as cfg
    from src.modeling.bert import BertConfig, Graphormer
    from src.modeling.bert import Graphormer_Hand_Network
    from src.modeling._mano import MANO, Mesh
    from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
    from src.modeling.hrnet.config import config as hrnet_config
    from src.modeling.hrnet.config import update_config as hrnet_update_config
    from src.utils.miscellaneous import set_seed
    from src.utils.geometric_layers import orthographic_projection
except ImportError as e:  # pragma: no cover
    print(f"Error importing MeshGraphormer components: {e}")
    print("Ensure MeshGraphormer is installed or PYTHONPATH is set correctly.")
    sys.exit(1)

# --------- PyTorch3D for depth rasterization & visualization ----------
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    FoVOrthographicCameras,
    MeshRenderer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
)

# --------- SAM2 + YOLO ----------
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

# --------- Optional Hungarian (SciPy) with fallback ----------
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def hungarian_or_greedy(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (row, col) matches minimizing cost.

    Uses SciPy's Hungarian algorithm if available, otherwise a simple greedy fallback.
    """
    if HAS_SCIPY:
        r, c = linear_sum_assignment(cost_matrix)
        return list(zip(r, c))

    # Greedy fallback
    pairs: List[Tuple[int, int]] = []
    cm = cost_matrix.copy()
    used_r: set[int] = set()
    used_c: set[int] = set()

    while True:
        try:
            finite_vals = cm[np.isfinite(cm)]
            if finite_vals.size == 0:
                break
            min_val = float(np.min(finite_vals))
            if min_val >= 1.0:
                # No useful match left (IoU too small)
                break

            coords = np.argwhere(cm == min_val)
            if coords.shape[0] == 0:
                break
            i, j = map(int, coords[0])

        except ValueError: # Handle empty cm
            break

        val = float(cm[i, j])
        if not np.isfinite(val) or val >= 1.0:
            cm[i, j] = np.inf
            continue

        if i in used_r or j in used_c:
            cm[i, j] = np.inf
            continue

        pairs.append((i, j))
        used_r.add(i)
        used_c.add(j)
        cm[i, :] = np.inf
        cm[:, j] = np.inf

    return pairs


# --------- TorchVision transforms for Graphormer input ----------
from torchvision import transforms

img_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
IMG_TRANSFORM_RESIZE = 224
IMG_TRANSFORM_CENTERCROP = 224

# ===============================
# Configuration / Constants
# ===============================
APP_NAME = "hand-fix-job"
APP_HOME = os.getcwd()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
ENABLE_FULL_DEBUG = os.environ.get("ENABLE_FULL_DEBUG", "false").lower() == "true"
DETERMINISTIC = os.environ.get("DETERMINISTIC", "false").lower() == "true"
QA_MVE_THRESHOLD = float(os.environ.get("QA_MVE_THRESHOLD", "15.0"))
MAX_RETRY_ATTEMPTS = int(os.environ.get("MAX_RETRY_ATTEMPTS", "2"))
TEMPORAL_SMOOTHING_ALPHA = float(os.environ.get("TEMPORAL_SMOOTHING_ALPHA", "0.5"))
MASK_DILATION_PIXELS = int(os.environ.get("MASK_DILATION_PIXELS", "10"))
MASK_DILATION_KERNEL = (
    np.ones((MASK_DILATION_PIXELS, MASK_DILATION_PIXELS), np.uint8)
    if MASK_DILATION_PIXELS > 0
    else None
)
MASK_FEATHER_SIGMA = float(os.environ.get("MASK_FEATHER_SIGMA", "1.0"))
MIN_HAND_AREA_PIXELS = int(os.environ.get("MIN_HAND_AREA_PIXELS", "2500"))

SDXL_PROMPT = os.environ.get(
    "SDXL_PROMPT",
    (
        "a hyper-realistic photo of a beautiful female human hand, "
        "perfect anatomy, 5 fingers, detailed skin texture, 8k, photorealistic"
    ),
)
SDXL_NEGATIVE_PROMPT = os.environ.get(
    "SDXL_NEGATIVE_PROMPT",
    (
        "blurry, disfigured, malformed, extra fingers, missing fingers, "
        "extra limbs, ugly, worst quality, low quality, normal quality, "
        "signature, watermark, text, letters"
    ),
)
SDXL_NUM_INFERENCE_STEPS = int(os.environ.get("SDXL_NUM_INFERENCE_STEPS", "35"))
SDXL_GUIDANCE_SCALE = float(os.environ.get("SDXL_GUIDANCE_SCALE", "8.0"))
SDXL_CONTROLNET_SCALE = float(os.environ.get("SDXL_CONTROLNET_SCALE", "0.75"))
SDXL_STRENGTH = float(os.environ.get("SDXL_STRENGTH", "0.95"))
SDXL_SEED = int(os.environ.get("SDXL_SEED", "42"))
SDXL_CPU_OFFLOAD_ENABLED = os.environ.get("SDXL_CPU_OFFLOAD", "true").lower() == "true"

YOLO_CONF_SAM = float(os.environ.get("YOLO_CONF_SAM", "0.5"))
YOLO_CONF_CROP = float(os.environ.get("YOLO_CONF_CROP", "0.25"))
SAM2_WARN_FRAMES = int(os.environ.get("SAM2_WARN_FRAMES", "900"))

CONTROLNET_CHECKPOINT = "diffusers/controlnet-depth-sdxl-1.0"
SDXL_INPAINT_CHECKPOINT = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
VAE_CHECKPOINT = "madebyollin/sdxl-vae-fp16-fix"
YOLO_CHECKPOINT = "hand_yolov8s.pt"

# --- [MODIFIED] SAM2 paths ---
# We now download the entire SAM2 GCS folder to SAM2_MODEL_ROOT_LOCAL
# All internal paths are relative to this new root.
SAM2_MODEL_ROOT_LOCAL = os.path.join(APP_HOME, "sam2_models")

SAM2_CHECKPOINT_NAME = "sam2.1_hiera_large.pt"
SAM2_LOCAL_DIR = SAM2_MODEL_ROOT_LOCAL # CHANGED: Was /app/checkpoints
SAM2_CHECKPOINT = os.path.join(SAM2_LOCAL_DIR, SAM2_CHECKPOINT_NAME) # Now /app/sam2_models/sam2.1_hiera_large.pt

SAM2_CONFIG_NAME_TEMPLATE = "sam2.1_hiera_{size}.yaml" # Template
# --- End SAM2 paths ---


# --- MeshGraphormer paths ---
MESHGRAPHORMER_CHECKPOINT_NAME = "graphormer_hand_state_dict.bin"
MESHGRAPHORMER_CHECKPOINT_DIR = os.path.join(APP_HOME, "MeshGraphormer/models/graphormer_release")
MESHGRAPHORMER_CHECKPOINT = os.path.join(MESHGRAPHORMER_CHECKPOINT_DIR, MESHGRAPHORMER_CHECKPOINT_NAME)

# --- CORRECTED DATA PATHS ---
# This is the hardcoded relative path MeshGraphormer looks for: 'src/modeling/data'
MESHGRAPHORMER_DATA_DIR = os.path.join(APP_HOME, "src", "modeling", "data")
YOLO_CHECKPOINT_PATH = os.path.join(APP_HOME, YOLO_CHECKPOINT)

MANO_LEFT_NAME = "MANO_LEFT.pkl"
MANO_RIGHT_NAME = "MANO_RIGHT.pkl"
MANO_SAMPLING_MATRIX_NAME = "mano_downsampling.npz"
SMPL_SAMPLING_MATRIX_NAME = "mesh_downsampling.npz"
JOINT_REGRESSOR_TRAIN_EXTRA_NAME = "J_regressor_extra.npy"
JOINT_REGRESSOR_H36M_CORRECT_NAME = "J_regressor_h36m_correct.npy"
SMPL_FILE_NAME = "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
SMPL_MALE_NAME = "basicModel_m_lbs_10_207_0_v1.0.0.pkl"
SMPL_FEMALE_NAME = "basicModel_f_lbs_10_207_0_v1.0.0.pkl"

MANO_LEFT_PATH = os.path.join(MESHGRAPHORMER_DATA_DIR, MANO_LEFT_NAME)
MANO_RIGHT_PATH = os.path.join(MESHGRAPHORMER_DATA_DIR, MANO_RIGHT_NAME)
MANO_SAMPLING_MATRIX_PATH = os.path.join(MESHGRAPHORMER_DATA_DIR, MANO_SAMPLING_MATRIX_NAME)
SMPL_SAMPLING_MATRIX_PATH = os.path.join(MESHGRAPHORMER_DATA_DIR, SMPL_SAMPLING_MATRIX_NAME)
JOINT_REGRESSOR_TRAIN_EXTRA_PATH = os.path.join(MESHGRAPHORMER_DATA_DIR, JOINT_REGRESSOR_TRAIN_EXTRA_NAME)
JOINT_REGRESSOR_H36M_CORRECT_PATH = os.path.join(MESHGRAPHORMER_DATA_DIR, JOINT_REGRESSOR_H36M_CORRECT_NAME)
SMPL_FILE_PATH = os.path.join(MESHGRAPHORMER_DATA_DIR, SMPL_FILE_NAME)
SMPL_MALE_PATH = os.path.join(MESHGRAPHORMER_DATA_DIR, SMPL_MALE_NAME)
SMPL_FEMALE_PATH = os.path.join(MESHGRAPHORMER_DATA_DIR, SMPL_FEMALE_NAME)
# --- End data paths ---

# --- HRNet paths ---
HRNET_CHECKPOINT_NAME = "hrnetv2_w64_imagenet_pretrained.pth"
HRNET_CONFIG_NAME = "cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
# --- CORRECTED HRNET PATH: Point inside the cloned repo ---
HRNET_LOCAL_DIR = os.path.join(APP_HOME, "MeshGraphormer", "models", "hrnet")
HRNET_CHECKPOINT = os.path.join(HRNET_LOCAL_DIR, HRNET_CHECKPOINT_NAME)
HRNET_CONFIG_YAML = os.path.join(HRNET_LOCAL_DIR, HRNET_CONFIG_NAME)
# --- END CORRECTION ---

BERT_CONFIG_PATH = os.path.join(APP_HOME, "MeshGraphormer", "src", "modeling", "bert", "bert-base-uncased")

# --- Local Dirs ---
TEMP_DIR = "/tmp/hand-fix"
DRIVING_VIDEO_LOCAL = os.path.join(TEMP_DIR, "driving_video.mp4")
GEN_VIDEO_LOCAL = os.path.join(TEMP_DIR, "generated_video.mp4")
DRIVING_FRAMES_DIR = os.path.join(TEMP_DIR, "driving_frames")
DRIVING_MESH_DIR = os.path.join(TEMP_DIR, "driving_mesh_data_v10")
TARGET_DEPTH_DIR = os.path.join(TEMP_DIR, "target_depth_maps_scaled")
GEN_FRAMES_DIR = os.path.join(TEMP_DIR, "generated_frames")
GEN_MESH_DIR = os.path.join(TEMP_DIR, "gen_mesh_data_v10")
MASKS_DIR = os.path.join(TEMP_DIR, "masks")
RETRY_MASKS_DIR = os.path.join(TEMP_DIR, "masks_retry")
FIXED_FRAMES_DIR = os.path.join(TEMP_DIR, "fixed_frames")
QA_MESH_DIR = os.path.join(TEMP_DIR, "qa_mesh_data_v10")
DIAGNOSTICS_DIR = os.path.join(TEMP_DIR, "diagnostics")
FRAME_DIAGNOSTICS_DIR = os.path.join(TEMP_DIR, "frame_diagnostics")


# ===============================
# Logging
# ===============================
def setup_logging() -> logging.Logger:
    logger_instance = logging.getLogger()
    log_level_setting = "DEBUG" if ENABLE_FULL_DEBUG else LOG_LEVEL

    for h in list(logger_instance.handlers):
        logger_instance.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt=f"%(asctime)s - {APP_NAME} - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger_instance.addHandler(handler)
    logger_instance.setLevel(log_level_setting)

    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("google.cloud").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("pytorch3d").setLevel(logging.WARNING)
    logging.getLogger("src").setLevel(logging.INFO)

    if ENABLE_FULL_DEBUG:
        logger_instance.info("--- FULL DEBUG MODE ENABLED ---")
        logger_instance.setLevel(logging.DEBUG)

    return logger_instance


logger = setup_logging()

# Torch global perf tweaks
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = not DETERMINISTIC
    torch.backends.cudnn.deterministic = DETERMINISTIC


# ===============================
# GCS Helpers
# ===============================
def parse_gcs_path(gcs_path: str) -> Tuple[str, str]:
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""
    return bucket_name, blob_name


def download_from_gcs(gcs_path: str, local_path: str) -> bool:
    logger.info("Downloading %s to %s", gcs_path, local_path)
    try:
        storage_client = storage.Client()
        bucket_name, blob_name = parse_gcs_path(gcs_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info("Successfully downloaded %s", local_path)
        return True
    except Exception as e:  # pragma: no cover
        logger.error("Failed to download %s to %s: %s", gcs_path, local_path, e)
        return False


def download_gcs_folder(gcs_folder_path: str, local_dir: str) -> bool:
    """Downloads all files from a GCS prefix (folder) to a local directory."""
    logger.info("Recursively downloading %s to %s", gcs_folder_path, local_dir)
    try:
        storage_client = storage.Client()
        bucket_name, prefix = parse_gcs_path(gcs_folder_path)
        if not prefix.endswith('/'):
            prefix += '/'
            
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            logger.warning("No files found in GCS folder: %s", gcs_folder_path)
            return True 

        file_count = 0
        for blob in blobs:
            if blob.name.endswith('/'):
                continue

            relative_path = os.path.relpath(blob.name, prefix)
            local_path = os.path.join(local_dir, relative_path)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            logger.debug("Downloading %s to %s", blob.name, local_path)
            blob.download_to_filename(local_path)
            file_count += 1

        logger.info("Successfully downloaded %d files from %s", file_count, gcs_folder_path)
        return True
    except Exception as e:  # pragma: no cover
        logger.error("Failed to download folder %s: %s", gcs_folder_path, e)
        return False


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    logger.info("Uploading %s to %s", local_path, gcs_path)
    storage_client = storage.Client()
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    bucket = storage_client.bucket(bucket_name)

    if os.path.isdir(local_path):
        for local_file in glob.glob(os.path.join(local_path, "**", "*"), recursive=True):
            if os.path.isfile(local_file):
                remote_path = os.path.join(
                    blob_name, os.path.relpath(local_file, local_path)
                )
                blob_up = bucket.blob(remote_path)
                blob_up.upload_from_filename(local_file)
    elif os.path.isfile(local_path):
        blob_up = bucket.blob(blob_name)
        blob_up.upload_from_filename(local_path)
    else:
        logger.warning("Local path %s not found for upload.", local_path)


# ===============================
# Video helpers
# ===============================
def get_video_metadata(video_path: str) -> Tuple[int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if width == 0 or height == 0:
        raise IOError(f"Bad metadata: {video_path}")
    return width, height, fps


def extract_frames(video_path: str, out_dir: str) -> List[str]:
    logger.info("Extracting frames from %s to %s", video_path, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    paths: List[str] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f"{idx:06d}.png")
        cv2.imwrite(out_path, frame)
        paths.append(out_path)
        idx += 1
    cap.release()

    logger.info("Extracted %d frames.", idx)
    return paths


def frames_to_video(frames_dir: str, out_path: str, fps: float) -> None:
    logger.info("Compiling frames from %s to %s at %.4g FPS", frames_dir, out_path, fps)

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames in {frames_dir}")
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise IOError(f"Cannot read: {frame_paths[0]}")

    H, W, _ = first.shape
    W = max(2, W if W % 2 == 0 else W - 1)
    H = max(2, H if H % 2 == 0 else H - 1)

    cmd = [
        "ffmpeg", "-y", "-r", str(fps), "-i", f"{frames_dir}/%06d.png",
        "-vf", f"scale={W}:{H}", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "18", out_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    logger.info("Video compiled: %s", out_path)


# ===============================
# Alignment (Umeyama: s, R, t)
# ===============================
def umeyama_align(
    X: np.ndarray, Y: np.ndarray, with_scale: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    muX, muY = X.mean(axis=0), Y.mean(axis=0)
    X0, Y0 = X - muX, Y - muY
    C = (X0.T @ Y0) / X.shape[0]
    U, S, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    if with_scale:
        varX = (X0 ** 2).sum() / X.shape[0]
        s = float(S.sum() / max(varX, 1e-12))
    else:
        s = 1.0

    t = muY - s * (R @ muX)
    return float(s), R, t


# ===============================
# Core pipeline
# ===============================
class HandFixPipeline:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.yolo_model: Optional[YOLO] = None
        self.sam_model: Optional[Any] = None
        self.graphormer_model: Optional[torch.nn.Module] = None
        self.mano_model: Optional[MANO] = None
        self.mesh_sampler: Optional[Mesh] = None
        self.sdxl_pipeline: Optional[StableDiffusionXLControlNetInpaintPipeline] = None
        self.enable_full_debug = ENABLE_FULL_DEBUG

        self.mesh_renderer_debug: Optional[MeshRenderer] = None
        self.debug_lights: Optional[PointLights] = None
        self.debug_shader: Optional[SoftPhongShader] = None
        self.debug_cameras: Optional[FoVOrthographicCameras] = None

        n_gpu = torch.cuda.device_count() if self.device.type == "cuda" else 0
        logger.debug("Setting seed %d with n_gpu=%d", SDXL_SEED, n_gpu)
        set_seed(SDXL_SEED, n_gpu)


    # ---------- Model Loading ----------
    def load_models(self) -> None:
        logger.info("Loading models onto device: %s", self.device)
        logger.info("SciPy (Hungarian) available: %s", HAS_SCIPY)

        # --- MeshGraphormer stack ---
        logger.info("Loading MeshGraphormer model...")
        try:
            args = SimpleNamespace(
                num_workers=4,
                img_scale_factor=1,
                image_file_or_path="./samples/hand",
                model_name_or_path=BERT_CONFIG_PATH,
                resume_checkpoint=MESHGRAPHORMER_CHECKPOINT,
                output_dir="output/",
                config_name="",
                arch="hrnet-w64",
                num_hidden_layers=4,
                hidden_size=-1,
                num_attention_heads=4,
                intermediate_size=-1,
                input_feat_dim="2051,512,128",
                hidden_feat_dim="1024,256,64",
                which_gcn="0,0,1",
                mesh_type="hand",
                run_eval_only=True,
                device=self.device,
                seed=88,
                num_gpus=1,
            )

            # --- Config overrides are NO LONGER NEEDED ---
            # We are downloading files to the default paths in main()

            # Internal MANO + Mesh sampler
            # These classes will now find their files at the default paths
            self.mano_model = MANO().to(args.device)
            self.mesh_sampler = Mesh()

            logger.debug("Loading HRNet backbone...")
            # HRNET_CONFIG_YAML points to the downloaded file
            if not os.path.exists(HRNET_CONFIG_YAML):
                raise FileNotFoundError(f"HRNet config not found at {HRNET_CONFIG_YAML}")
            hrnet_update_config(hrnet_config, HRNET_CONFIG_YAML) 
            
            hrnet_pretrained = HRNET_CHECKPOINT if os.path.exists(HRNET_CHECKPOINT) else ""
            if not hrnet_pretrained:
                logger.warning("HRNet checkpoint %s not found!", HRNET_CHECKPOINT)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_pretrained)
            logger.info("=> loaded HRNet backbone: %s", args.arch)

            logger.debug("Building Graphormer transformer blocks...")
            trans_encoder: List[torch.nn.Module] = []
            input_feat_dim = [int(x) for x in args.input_feat_dim.split(",")]
            hidden_feat_dim = [int(x) for x in args.hidden_feat_dim.split(",")]
            output_feat_dim = input_feat_dim[1:] + [3]
            which_blk_graph = [int(x) for x in args.which_gcn.split(",")]

            for i, out_dim in enumerate(output_feat_dim):
                config_class, model_class = BertConfig, Graphormer
                local_cfg_path = os.path.join(args.model_name_or_path, "config.json")
                bert_source = (
                    args.model_name_or_path
                    if os.path.exists(local_cfg_path)
                    else "bert-base-uncased"
                )
                logger.debug("Block %d: BERT source: %s", i, bert_source)

                trans_config = config_class.from_pretrained(bert_source)
                trans_config.output_attentions = False
                trans_config.img_feature_dim = input_feat_dim[i]
                trans_config.output_feature_dim = out_dim
                cur_hidden = hidden_feat_dim[i]
                trans_config.num_hidden_layers = args.num_hidden_layers
                trans_config.hidden_size = cur_hidden
                trans_config.num_attention_heads = args.num_attention_heads
                trans_config.intermediate_size = int(cur_hidden * 2)
                trans_config.graph_conv = which_blk_graph[i] == 1
                trans_config.mesh_type = args.mesh_type
                assert trans_config.hidden_size % trans_config.num_attention_heads == 0

                trans_encoder.append(model_class(config=trans_config))

            trans_encoder_seq = torch.nn.Sequential(*trans_encoder)

            cfg.output_attentions = False

            self.graphormer_model = Graphormer_Hand_Network(
                args, cfg, backbone, trans_encoder_seq
            ).to(args.device)

            logger.info("Loading Graphormer checkpoint: %s", args.resume_checkpoint)
            if not os.path.exists(args.resume_checkpoint):
                raise FileNotFoundError(f"Checkpoint missing: {args.resume_checkpoint}")

            checkpoint = torch.load(args.resume_checkpoint, map_location=args.device)
            loaded_state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            load_result = self.graphormer_model.load_state_dict(
                loaded_state_dict, strict=False
            )
            logger.info(
                "Graphormer Load: Missing=%s, Unexpected=%s",
                load_result.missing_keys,
                load_result.unexpected_keys,
            )
            self.graphormer_model.eval()
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to load MeshGraphormer model: %s", e)
            raise

        # --- YOLO ---
        logger.info("Loading YOLO model: %s", YOLO_CHECKPOINT)
        self.yolo_model = YOLO(YOLO_CHECKPOINT).to(self.device)

        # --- [FINAL SAM2 FIX] ---
        logger.info("Building SAM2 video predictor: %s", SAM2_CHECKPOINT)
        try:
            ckpt_lower = SAM2_CHECKPOINT_NAME.lower()
            if "hiera_large" in ckpt_lower: model_size = "l"
            elif "hiera_base" in ckpt_lower: model_size = "b"
            elif "hiera_tiny" in ckpt_lower: model_size = "t"
            else: model_size = "l"; logger.warning("Guessing SAM2 model size 'large'.")

            sam2_config_file_name = SAM2_CONFIG_NAME_TEMPLATE.format(size=model_size)
            
            # --- Path definitions ---
            sam2_checkpoint_path = SAM2_CHECKPOINT
            sam2_config_path_absolute = os.path.join(SAM2_MODEL_ROOT_LOCAL, sam2_config_file_name)
            
            if not os.path.exists(sam2_checkpoint_path):
                raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_checkpoint_path}")

            if not os.path.exists(sam2_config_path_absolute):
                raise FileNotFoundError(f"SAM2 config yaml not found at: {sam2_config_path_absolute}")

            # --- Hydra Fix v2: Add config dir to search path ---
            # We must initialize Hydra *before* calling compose.
            # We point it to the directory containing the config.
            try:
                GlobalHydra.instance().clear() # Clear any previous state
                # initialize_config_dir requires an absolute path
                logger.debug("Initializing Hydra with config_dir: %s", SAM2_MODEL_ROOT_LOCAL)
                initialize_config_dir(config_dir=SAM2_MODEL_ROOT_LOCAL, version_base=None)
            except Exception as e:
                # This might fail if it's already initialized, which is fine.
                logger.warning("Hydra re-initialization warning: %s", e)

            logger.debug("Building SAM2 with config_name: %s", sam2_config_file_name)
            self.sam_model = build_sam2_video_predictor(
                sam2_config_file_name,  # Pass FILENAME ONLY
                sam2_checkpoint_path
            )
            # --- [END SAM2 FIX v2] ---
                
            self.sam_model.to(self.device)
            logger.info("SAM2 video predictor built successfully.")
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to build SAM2 video predictor: %s", e)
            raise
        # --- [END SAM2 FIX] ---

        # --- SDXL + ControlNet ---
        logger.info("Loading ControlNet: %s", CONTROLNET_CHECKPOINT)
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_CHECKPOINT, torch_dtype=self.torch_dtype
        )
        logger.info("Loading VAE: %s", VAE_CHECKPOINT)
        vae = AutoencoderKL.from_pretrained(VAE_CHECKPOINT, torch_dtype=self.torch_dtype)
        logger.info("Loading SDXL Pipeline: %s", SDXL_INPAINT_CHECKPOINT)

        if self.torch_dtype == torch.float16:
            self.sdxl_pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                SDXL_INPAINT_CHECKPOINT, controlnet=controlnet, vae=vae,
                torch_dtype=self.torch_dtype, variant="fp16", use_safetensors=True
            )
        else:
            self.sdxl_pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                SDXL_INPAINT_CHECKPOINT, controlnet=controlnet, vae=vae,
                torch_dtype=self.torch_dtype, use_safetensors=True
            )

        if SDXL_CPU_OFFLOAD_ENABLED:
            logger.info("Enabling SDXL CPU Offload.")
            try:
                self.sdxl_pipeline.enable_model_cpu_offload()
            except Exception as e:  # pragma: no cover
                logger.warning("CPU offload failed (%s).", e)
                self.sdxl_pipeline.to(self.device, dtype=self.torch_dtype)
        else:
            logger.info("SDXL CPU Offload DISABLED.")
            self.sdxl_pipeline.to(self.device, dtype=self.torch_dtype)
        try:
            self.sdxl_pipeline.enable_vae_slicing()
            self.sdxl_pipeline.enable_vae_tiling()
            self.sdxl_pipeline.enable_attention_slicing("max")
            logger.info("Enabled SDXL memory savers.")
        except Exception as e:  # pragma: no cover
            logger.warning("Could not enable diffusion memory savers: %s", e)

        # --- Debug Renderer ---
        if self.enable_full_debug:
            logger.debug("Initializing PyTorch3D debug renderer...")
            self.debug_cameras = FoVOrthographicCameras(device=self.device)
            self.debug_lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
            self.debug_shader = SoftPhongShader(
                device=self.device, cameras=self.debug_cameras, lights=self.debug_lights
            )
            self.mesh_renderer_debug = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.debug_cameras,
                    raster_settings=RasterizationSettings(
                        image_size=512, blur_radius=0.0, faces_per_pixel=1
                    )
                ),
                shader=self.debug_shader
            )
            logger.debug("Debug renderer initialized.")

        logger.info("All models loaded successfully.")

    # --- BBox IoU & helpers ---
    @staticmethod
    def _calculate_iou(boxA: List[int], boxB: List[int]) -> float:
        xA=max(boxA[0],boxB[0]);yA=max(boxA[1],boxB[1]);xB=min(boxA[2],boxB[2]);yB=min(boxA[3],boxB[3])
        interArea=max(0,xB-xA)*max(0,yB-yA);boxAArea=max(0,boxA[2]-boxA[0])*max(0,boxA[3]-boxA[1])
        boxBArea=max(0,boxB[2]-boxB[0])*max(0,boxB[3]-boxB[1]);unionArea=float(boxAArea+boxBArea-interArea)
        return interArea/unionArea if unionArea>0 else 0.0

    @staticmethod
    def _get_hand_components(mask_gray: np.ndarray, min_area: int = 1000) -> List[List[int]]:
        bboxes: List[List[int]] = []
        if mask_gray is None or not mask_gray.any(): return bboxes
        cnts, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return bboxes
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                bboxes.append([x, y, x + w, y + h])
        return bboxes

    @staticmethod
    def _filter_small_boxes(boxes: np.ndarray, min_area: int = MIN_HAND_AREA_PIXELS) -> List[List[int]]:
        keep: List[List[int]] = []
        for b in boxes:
            x0, y0, x1, y1 = map(int, b)
            if (x1 - x0) * (y1 - y0) >= min_area:
                keep.append([x0, y0, x1, y1])
        return keep

    @staticmethod
    def _clamp_bbox(bbox: List[int], W: int, H: int) -> List[int]:
        x0, y0, x1, y1 = map(int, bbox)
        return [max(0, x0), max(0, y0), min(W, x1), min(H, y1)]

    def _crop_image_by_bbox(self, img_bgr: np.ndarray, bbox: List[int], pad: int = 16) -> np.ndarray:
        H, W = img_bgr.shape[:2]; x0, y0, x1, y1 = bbox
        x0=max(0,x0-pad); y0=max(0,y0-pad); x1=min(W,x1+pad); y1=min(H,y1+pad)
        x0, y0, x1, y1 = self._clamp_bbox([x0, y0, x1, y1], W, H)
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            s = int(min(H, W) * 0.6); x0_c = (W - s) // 2; y0_c = (H - s) // 2
            return img_bgr[y0_c : y0_c + s, x0_c : x0_c + s]
        return crop

    def _get_hand_bboxes_and_crops(
        self, img_bgr: np.ndarray, mask_gray: Optional[np.ndarray] = None, pad: int = 16
    ) -> List[Tuple[np.ndarray, List[int]]]:
        hand_bboxes: List[List[int]] = []
        H, W = img_bgr.shape[:2]
        if mask_gray is not None and mask_gray.any():
            hand_bboxes = self._get_hand_components(mask_gray, min_area=MIN_HAND_AREA_PIXELS)
            logger.debug("Found %d hands from mask.", len(hand_bboxes))
        if not hand_bboxes:
            yolo_bboxes_np = self._detect_hand_bboxes(img_bgr, conf=YOLO_CONF_CROP)
            if yolo_bboxes_np.shape[0] > 0:
                hand_bboxes = self._filter_small_boxes(yolo_bboxes_np)
                logger.debug("Found %d hands from YOLO.", len(hand_bboxes))
            else:
                logger.debug("No hands found.")
        crops_and_bboxes: List[Tuple[np.ndarray, List[int]]] = []
        for bbox in hand_bboxes:
            bbox = self._clamp_bbox(bbox, W, H)
            crop = self._crop_image_by_bbox(img_bgr, bbox, pad=pad)
            crops_and_bboxes.append((crop, bbox))
        return crops_and_bboxes

    def _detect_hand_bboxes(self, img_bgr: np.ndarray, conf: float = 0.3) -> np.ndarray:
        try:
            res = self.yolo_model(img_bgr, conf=conf, verbose=False)
            boxes = res[0].boxes.xyxy.cpu().numpy() if len(res) else np.zeros((0, 4))
            return boxes
        except Exception as e:  # pragma: no cover
            logger.warning("YOLO failed: %s", e)
            return np.zeros((0, 4))

    # ---------- SAM2 masks ----------
    def get_hand_masks_sam(
        self, gen_video_path: str, frame_paths: List[str], out_dir: str
    ) -> bool:
        """
        Generates masks using SAM2 stateful video predictor.
        """
        os.makedirs(out_dir, exist_ok=True)
        logger.info("Starting SAM2 mask generation for %d frames...", len(frame_paths))
        if len(frame_paths) > SAM2_WARN_FRAMES:
            logger.warning("SAM2 processing %d frames; may be memory intensive.", len(frame_paths))

        # --- Load first frame manually for YOLO ---
        try:
            first_bgr = cv2.imread(frame_paths[0])
            if first_bgr is None:
                raise IOError(f"Could not read first frame: {frame_paths[0]}")
            H, W, _ = first_bgr.shape
        except Exception as e:
            logger.error("Failed to read first frame for YOLO: %s", e)
            return False
        
        yolo_results = self.yolo_model(first_bgr, conf=YOLO_CONF_SAM, verbose=False)
        bboxes = yolo_results[0].boxes.xyxy.cpu().numpy() if len(yolo_results) else np.zeros((0, 4))
        bboxes = self._filter_small_boxes(bboxes)
        logger.debug("SAM2 using %d filtered prompt boxes from frame 0.", len(bboxes))

        if len(bboxes) == 0:
            logger.warning("No hands on frame 0 (YOLO). Falling back to per-frame YOLO masks.")
            return self._fallback_yolo_rect_masks(frame_paths, out_dir)

        logger.info("Running SAM2 video segmentation with %d prompt boxes...", len(bboxes))
        try:
            autocast_ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else torch.no_grad()
            )

            with torch.inference_mode(), autocast_ctx:
                # --- Use init_state with VIDEO PATH ---
                logger.debug("Initializing SAM2 state with video: %s", gen_video_path)
                state = self.sam_model.init_state(video_path=gen_video_path)

                initial_prompts = {0: {"box": bboxes.tolist()}}
                logger.debug("Adding initial prompts: %s", initial_prompts)
                self.sam_model.add_new_prompts(state, initial_prompts)
                logger.debug("Initial prompts added.")

                temp_masks: Dict[int, torch.Tensor] = {}
                logger.debug("Propagating SAM2 masks...")

                for frame_idx, _object_ids, masks in self.sam_model.propagate_in_video(state):
                    if masks is not None and masks.numel() > 0:
                        frame_mask = torch.any(masks, dim=0)  # [H, W] bool
                        temp_masks[int(frame_idx)] = frame_mask
                    else:
                        logger.debug("No mask found for frame %d", frame_idx)

                # Save masks in order, filling gaps
                saved_count = 0
                num_frames = len(frame_paths)
                for idx in range(num_frames):
                    if idx in temp_masks:
                        mask_hw = temp_masks[idx].cpu().numpy().astype(np.uint8) * 255
                    else:
                        mask_hw = np.zeros((H, W), dtype=np.uint8) # Use H,W from first frame

                    save_path = os.path.join(out_dir, f"{idx:06d}.png")
                    cv2.imwrite(save_path, mask_hw)
                    saved_count += 1

                logger.info("Mask saving complete for %d frames.", saved_count)
                return True
        except Exception as e:  # pragma: no cover
            logger.exception("SAM2 failed during processing: %s. Falling back...", e)
            return self._fallback_yolo_rect_masks(frame_paths, out_dir)

    def _fallback_yolo_rect_masks(self, frame_paths: List[str], out_dir: str) -> bool:
        os.makedirs(out_dir, exist_ok=True)
        ok = True
        for i, p in enumerate(tqdm(frame_paths, desc="YOLO Masks")):
            img = cv2.imread(p)
            if img is None:
                ok = False; continue
            H, W = img.shape[:2]
            boxes = self._detect_hand_bboxes(img, conf=YOLO_CONF_CROP)
            boxes = self._filter_small_boxes(boxes)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for box in boxes:
                x0, y0, x1, y1 = self._clamp_bbox(box, W, H)
                cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
            if MASK_DILATION_KERNEL is not None:
                mask = cv2.dilate(mask, MASK_DILATION_KERNEL, iterations=1)
            cv2.imwrite(os.path.join(out_dir, f"{i:06d}.png"), mask)
        return ok

    # ---------- MeshGraphormer ----------
    def get_hand_mesh_data(
        self, frame_paths: List[str], out_dir: str, masks_dir: Optional[str] = None
    ) -> List[List[Dict[str, Any]]]:
        logger.info("Running MeshGraphormer on %d frames...", len(frame_paths))
        os.makedirs(out_dir, exist_ok=True)
        all_frames_data: List[List[Dict[str, Any]]] = []

        for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Mesh Extraction")):
            frame_hand_data: List[Dict[str, Any]] = []
            save_data: Dict[str, Any] = {}
            debug_frame_dir = os.path.join(FRAME_DIAGNOSTICS_DIR, f"frame_{frame_idx:06d}")
            try:
                img_bgr = cv2.imread(frame_path)
                if img_bgr is None: raise IOError(f"Cannot read: {frame_path}")
                mask_gray: Optional[np.ndarray] = None
                if masks_dir:
                    mask_path = os.path.join(
                        masks_dir, f"{os.path.splitext(os.path.basename(frame_path))[0]}.png"
                    )
                    if os.path.exists(mask_path):
                        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        logger.debug("[F%06d] Mask loaded", frame_idx)
                    else:
                        logger.debug("[F%06d] Mask not found", frame_idx)

                crops_and_bboxes = self._get_hand_bboxes_and_crops(img_bgr, mask_gray)
                logger.debug("[F%06d] Found %d hands", frame_idx, len(crops_and_bboxes))

                if self.enable_full_debug and crops_and_bboxes:
                    os.makedirs(debug_frame_dir, exist_ok=True)
                    img_with_boxes = img_bgr.copy()
                    for hand_idx, (_crop_bgr, bbox) in enumerate(crops_and_bboxes):
                        x0, y0, x1, y1 = bbox
                        cv2.rectangle(img_with_boxes, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.putText(img_with_boxes, f"h_{hand_idx}", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    save_path = os.path.join(debug_frame_dir, f"f{frame_idx:06d}_01_box.png")
                    cv2.imwrite(save_path, img_with_boxes)

                for hand_idx, (crop_bgr, bbox) in enumerate(crops_and_bboxes):
                    logger.debug("[F%06d H%d] Box %s...", frame_idx, hand_idx, str(bbox))
                    img_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                    img_tensor = img_transform(img_pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = self.graphormer_model(
                            img_tensor, self.mano_model, self.mesh_sampler, is_train=False
                        )
                        if len(outputs) == 6: pred_camera, _, _, pred_vertices, _, _ = outputs
                        elif len(outputs) == 4: pred_camera, _, _, pred_vertices = outputs
                        else: raise ValueError(f"Bad outputs len: {len(outputs)}")
                    verts = pred_vertices.squeeze(0).cpu().numpy()
                    cam = pred_camera.squeeze(0).cpu().numpy()
                    logger.debug("[F%06d H%d] Mesh OK", frame_idx, hand_idx)
                    data = {'pred_vertices': verts, 'pred_camera': cam, 'bbox': np.array(bbox)}
                    frame_hand_data.append(data)
                    save_data[f'h{hand_idx}_v'] = verts
                    save_data[f'h{hand_idx}_c'] = cam
                    save_data[f'h{hand_idx}_b'] = np.array(bbox)

                base = os.path.splitext(os.path.basename(frame_path))[0]
                if save_data:
                    np.savez_compressed(os.path.join(out_dir, f"{base}_mesh.npz"), **save_data)
                all_frames_data.append(frame_hand_data)
            except Exception as e:  # pragma: no cover
                logger.exception("[F%06d] Mesh failed: %s", frame_idx, e)
                all_frames_data.append([])

        valid_frames = sum(1 for f in all_frames_data if f)
        total_hands = sum(len(f) for f in all_frames_data)
        logger.info("Mesh OK. Got %d hands across %d frames.", total_hands, valid_frames)
        return all_frames_data

    # ---------- Align meshes ----------
    def align_meshes_and_cameras(
        self,
        driving_mesh_data: List[List[Dict[str, Any]]],
        gen_mesh_data: List[List[Dict[str, Any]]],
        min_iou: float = 0.3,
    ) -> List[List[Dict[str, Any]]]:
        logger.info("Aligning meshes (multi-hand w/ IoU)...")
        if len(driving_mesh_data) != len(gen_mesh_data):
            logger.error("Mesh length mismatch D=%d, G=%d", len(driving_mesh_data), len(gen_mesh_data))
            min_len = min(len(driving_mesh_data), len(gen_mesh_data))
            driving_mesh_data = driving_mesh_data[:min_len]
            gen_mesh_data = gen_mesh_data[:min_len]

        aligned_frames: List[List[Dict[str, Any]]] = []
        total_matches = 0

        for frame_idx, (driv_hands, gen_hands) in enumerate(zip(driving_mesh_data, gen_mesh_data)):
            logger.debug("[F%06d] Aligning %d driv -> %d gen.", frame_idx, len(driv_hands), len(gen_hands))
            if not driv_hands or not gen_hands:
                aligned_frames.append([])
                continue

            driv_bboxes = [d["bbox"] for d in driv_hands]
            gen_bboxes = [g["bbox"] for g in gen_hands]
            cost_matrix = np.ones((len(driv_bboxes), len(gen_bboxes)), dtype=np.float64)
            for i, db in enumerate(driv_bboxes):
                for j, gb in enumerate(gen_bboxes):
                    iou = self._calculate_iou(db, gb)
                    cost_matrix[i, j] = 1.0 - iou
            logger.debug("[F%06d] Cost Matrix:\n%s", frame_idx, cost_matrix)

            pairs = hungarian_or_greedy(cost_matrix)
            aligned_hands: List[Dict[str, Any]] = []
            for i, j in pairs:
                iou_match = 1.0 - cost_matrix[i, j]
                if iou_match < min_iou:
                    logger.debug("[F%06d] Skip %d->%d IoU %.2f", frame_idx, i, j, iou_match)
                    continue
                logger.debug("[F%06d] Match %d->%d IoU %.2f", frame_idx, i, j, iou_match)
                try:
                    d = driv_hands[i]; g = gen_hands[j]
                    dV = d["pred_vertices"]; gV = g["pred_vertices"]
                    s, R, t = umeyama_align(dV, gV, with_scale=True)
                    logger.debug("[F%06d M%d->%d] Umeyama s=%.3f", frame_idx, i, j, s)
                    aligned_verts = (s * (dV @ R.T)) + t
                    cam = d["pred_camera"].copy()
                    cam_scale = cam[0]
                    cam[0] *= s
                    cam[1] += t[0] * cam_scale
                    cam[2] += t[1] * cam_scale
                    aligned_hands.append({
                        'aligned_verts': aligned_verts,
                        'aligned_cam': cam,
                        'bbox': g['bbox']
                    })
                    total_matches += 1
                except Exception as e:  # pragma: no cover
                    logger.error("[F%06d M%d->%d] Align failed: %s", frame_idx, i, j, e)
            aligned_frames.append(aligned_hands)

        logger.info("Mesh alignment OK. Found %d matches.", total_matches)
        return aligned_frames

    # --- Debug Overlay ---
    def _debug_overlay_mesh(
        self, mesh: Meshes, orig_frame_bgr: np.ndarray, H: int, W: int, out_path: str
    ) -> None:
        try:
            assert self.mesh_renderer_debug is not None
            self.mesh_renderer_debug.rasterizer.raster_settings.image_size = (H, W)
            rendered = (
                self.mesh_renderer_debug(mesh)[0, ..., :3].cpu().numpy() * 255.0
            ).astype(np.uint8)
            frags = self.mesh_renderer_debug.rasterizer(mesh)
            vis = (frags.pix_to_face[0] >= 0).cpu().numpy().astype(np.float32)[..., None]
            orig_rgb = cv2.cvtColor(orig_frame_bgr, cv2.COLOR_BGR2RGB)
            composite_rgb = (rendered * vis + orig_rgb * (1.0 - vis)).astype(np.uint8)
            composite_bgr = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, composite_bgr)
        except Exception as e:  # pragma: no cover
            logger.error("Failed debug overlay: %s", e)

    # ---------- Composite depth map ----------
    def render_depth_map(
        self,
        aligned_data_list: List[List[Dict[str, Any]]],
        gen_frame_paths: List[str],
        img_size: Tuple[int, int],
        out_dir: str,
    ) -> List[str]:
        logger.info("Rendering %d depth maps to %s...", len(aligned_data_list), out_dir)
        os.makedirs(out_dir, exist_ok=True)
        W, H = img_size
        cameras = self.debug_cameras if self.mesh_renderer_debug else FoVOrthographicCameras(device=self.device)
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, perspective_correct=False
            )
        )
        base_faces = torch.tensor(self.mano_model.face, dtype=torch.long, device=self.device)
        out_paths: List[str] = []
        failures = 0

        for i, frame_hands_data in enumerate(tqdm(aligned_data_list, desc="Rendering Depth")):
            out_path = os.path.join(out_dir, f"{i:06d}.png")
            debug_frame_dir = os.path.join(FRAME_DIAGNOSTICS_DIR, f"frame_{i:06d}")

            if not frame_hands_data:
                cv2.imwrite(out_path, np.zeros((H, W), np.uint8))
                out_paths.append(out_path); failures += 1
                continue

            all_verts_list: List[torch.Tensor] = []
            all_faces_list: List[torch.Tensor] = []
            num_verts_total = 0
            try:
                orig_frame_bgr = cv2.imread(gen_frame_paths[i])
                if orig_frame_bgr is None:
                    raise IOError(f"Cannot read: {gen_frame_paths[i]}")
                Hf, Wf = orig_frame_bgr.shape[:2]

                for hand_data in frame_hands_data:
                    verts3d = torch.tensor(hand_data["aligned_verts"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    cam = torch.tensor(hand_data["aligned_cam"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    bbox = self._clamp_bbox(hand_data["bbox"], Wf, Hf)

                    with torch.no_grad():
                        verts2d_proj = orthographic_projection(verts3d, cam).squeeze(0)

                    crop_bgr = orig_frame_bgr[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                    crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                    w, h = crop_pil.size
                    if w == 0 or h == 0:
                        logger.warning("[F%06d] Skip hand: zero crop %s", i, bbox)
                        continue

                    if w < h: new_w, new_h = IMG_TRANSFORM_RESIZE, int(IMG_TRANSFORM_RESIZE * h / w)
                    else: new_w, new_h = int(IMG_TRANSFORM_RESIZE * w / h), IMG_TRANSFORM_RESIZE

                    new_w = max(IMG_TRANSFORM_CENTERCROP, new_w)
                    new_h = max(IMG_TRANSFORM_CENTERCROP, new_h)
                    scale = (new_h / h) if w < h else (new_w / w)
                    x_offset = max(0.0, (new_w - IMG_TRANSFORM_CENTERCROP) / 2.0)
                    y_offset = max(0.0, (new_h - IMG_TRANSFORM_CENTERCROP) / 2.0)

                    verts2d_in_resized_crop = verts2d_proj + torch.tensor([x_offset, y_offset], device=self.device)
                    verts2d_in_orig_crop = verts2d_in_resized_crop / max(scale, 1e-6)
                    verts2d_in_full_frame = verts2d_in_orig_crop + torch.tensor([bbox[0], bbox[1]], device=self.device)

                    x_ndc = (verts2d_in_full_frame[:, 0] / (W / 2.0)) - 1.0
                    y_ndc = 1.0 - (verts2d_in_full_frame[:, 1] / (H / 2.0))
                    z = verts3d.squeeze(0)[:, 2]
                    z_ndc = (z - z.mean()) / (z.std() + 1e-6)
                    verts_ndc = torch.stack([x_ndc, y_ndc, z_ndc], dim=-1)
                    faces = base_faces + num_verts_total

                    all_verts_list.append(verts_ndc)
                    all_faces_list.append(faces)
                    num_verts_total += verts_ndc.shape[0]

                if not all_verts_list:
                    logger.warning("[F%06d] No valid hands for rasterization.", i)
                    cv2.imwrite(out_path, np.zeros((H, W), np.uint8))
                    out_paths.append(out_path); failures += 1
                    continue

                verts_cat = torch.cat(all_verts_list, dim=0).unsqueeze(0)
                faces_cat = torch.cat(all_faces_list, dim=0).unsqueeze(0)

                with torch.no_grad():
                    mesh = Meshes(verts=verts_cat, faces=faces_cat)
                    if self.mesh_renderer_debug:
                        os.makedirs(debug_frame_dir, exist_ok=True)
                        textures = TexturesVertex(
                            verts_features=torch.ones_like(verts_cat, device=self.device) *
                            torch.tensor([0.5, 0.5, 1.0], device=self.device)
                        )
                        mesh_vis = Meshes(verts=verts_cat, faces=faces_cat, textures=textures)
                        vis_path = os.path.join(debug_frame_dir, f"f{i:06d}_04_3dvis.png")
                        self._debug_overlay_mesh(mesh_vis, orig_frame_bgr, H, W, vis_path)

                    fragments = rasterizer(mesh)
                    zbuf = fragments.zbuf.squeeze(-1).squeeze(0)
                    valid = zbuf != -1
                    depth_img = torch.zeros((H, W), dtype=torch.float32, device=self.device)
                    if valid.any():
                        zmin, zmax = zbuf[valid].min(), zbuf[valid].max()
                        rng = (zmax - zmin).clamp(min=1e-6)
                        depth_norm = torch.clamp((zbuf - zmin) / rng, 0, 1)
                        depth_img[valid] = 1.0 - depth_norm[valid]

                depth_u8 = (depth_img.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(out_path, depth_u8)
                out_paths.append(out_path)

                if self.enable_full_debug:
                    os.makedirs(debug_frame_dir, exist_ok=True)
                    shutil.copy(out_path, os.path.join(debug_frame_dir, f"f{i:06d}_03_depth.png"))
            except Exception as e:  # pragma: no cover
                logger.exception("Depth render failed frame %d: %s", i, e)
                cv2.imwrite(out_path, np.zeros((H, W), np.uint8))
                out_paths.append(out_path)
                failures += 1

        logger.info("Saved %d depth maps (%d failures).", len(out_paths), failures)
        return out_paths

    # ---------- Guided inpainting ----------
    def run_guided_inpainting(
        self,
        gen_frame_paths: List[str],
        mask_frame_paths: List[str],
        driving_mesh_data: List[List[Dict[str, Any]]],
        gen_mesh_data: List[List[Dict[str, Any]]],
        out_dir: str,
        debug_upload_path_base: Optional[str] = None,
    ) -> Tuple[List[str], List[List[Dict[str, Any]]]]:

        logger.info("Starting inpainting for %d frames...", len(gen_frame_paths))
        os.makedirs(out_dir, exist_ok=True)

        aligned_data_list = self.align_meshes_and_cameras(driving_mesh_data, gen_mesh_data)

        sample = cv2.imread(gen_frame_paths[0])
        if sample is None:
            raise IOError(f"Cannot read: {gen_frame_paths[0]}")
        H, W = sample.shape[:2]

        depth_map_paths = self.render_depth_map(
            aligned_data_list, gen_frame_paths, (W, H), TARGET_DEPTH_DIR
        )

        gen_device = "cuda" if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(SDXL_SEED)

        prompt = SDXL_PROMPT
        neg_prompt = SDXL_NEGATIVE_PROMPT

        n = min(len(gen_frame_paths), len(mask_frame_paths), len(depth_map_paths))
        if n != len(gen_frame_paths):
            logger.warning("Frame count mismatch; processing %d frames.", n)

        prev_fixed: Optional[np.ndarray] = None
        prev_orig_gray: Optional[np.ndarray] = None
        flow_grid: Optional[Tuple[np.ndarray, np.ndarray]] = None
        out_paths: List[str] = []

        for i in tqdm(range(n), desc="Inpainting"):
            debug_frame_dir = os.path.join(FRAME_DIAGNOSTICS_DIR, f"frame_{i:06d}")
            try:
                init_cv = cv2.imread(gen_frame_paths[i])
                mask_gray = cv2.imread(mask_frame_paths[i], cv2.IMREAD_GRAYSCALE)
                depth_gray = cv2.imread(depth_map_paths[i], cv2.IMREAD_GRAYSCALE)
                if init_cv is None or mask_gray is None or depth_gray is None:
                    raise IOError(f"Read error @ frame {i}")

                logger.debug("[F%06d] Inpainting...", i)

                if MASK_FEATHER_SIGMA > 0:
                    mask_gray = cv2.GaussianBlur(
                        mask_gray, (0, 0), sigmaX=MASK_FEATHER_SIGMA
                    )

                if self.enable_full_debug:
                    os.makedirs(debug_frame_dir, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(debug_frame_dir, f"f{i:06d}_02_mask_fthr.png"), mask_gray
                    )

                init_pil = Image.fromarray(cv2.cvtColor(init_cv, cv2.COLOR_BGR2RGB))
                mask_pil = Image.fromarray(mask_gray).convert("L")
                control_pil = Image.fromarray(depth_gray).convert("L")

                if np.all(depth_gray == 0):
                    logger.warning("Depth map %d empty. Skipping ControlNet.", i)
                    output_pil = init_pil
                else:
                    logger.debug(
                        "[F%06d] SDXL Inpaint Str:%.3f, Scale:%.3f",
                        i, SDXL_STRENGTH, SDXL_CONTROLNET_SCALE
                    )
                    output_pil = self.sdxl_pipeline(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        image=init_pil,
                        mask_image=mask_pil,
                        control_image=control_pil,
                        num_inference_steps=SDXL_NUM_INFERENCE_STEPS,
                        strength=SDXL_STRENGTH,
                        guidance_scale=SDXL_GUIDANCE_SCALE,
                        controlnet_conditioning_scale=SDXL_CONTROLNET_SCALE,
                        generator=generator,
                    ).images[0]

                current_fixed = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
                final_out = current_fixed.copy()

                if (
                    i > 0
                    and prev_fixed is not None
                    and prev_orig_gray is not None
                    and TEMPORAL_SMOOTHING_ALPHA > 0
                ):
                    logger.debug("[F%06d] Temporal smooth α=%.3f...", i, TEMPORAL_SMOOTHING_ALPHA)
                    try:
                        cur_orig_gray = cv2.cvtColor(init_cv, cv2.COLOR_BGR2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_orig_gray, cur_orig_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        h, w = flow.shape[:2]
                        if flow_grid is None or flow_grid[0].shape[:2] != (h, w):
                            grid_y, grid_x = np.mgrid[0:h, 0:w]
                            flow_grid = (grid_x.astype(np.float32), grid_y.astype(np.float32))
                        
                        grid_x, grid_y = flow_grid
                        map_x = grid_x + flow[..., 0]
                        map_y = grid_y + flow[..., 1]
                        warped_prev = cv2.remap(prev_fixed, map_x, map_y, cv2.INTER_LINEAR)

                        _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
                        if MASK_FEATHER_SIGMA > 0:
                            mask_bin = cv2.GaussianBlur(mask_bin, (0, 0), sigmaX=MASK_FEATHER_SIGMA)
                        
                        mask_f = (mask_bin.astype(np.float32) / 255.0)[..., None]
                        alpha = float(TEMPORAL_SMOOTHING_ALPHA)
                        blended = cv2.addWeighted(current_fixed, 1.0 - alpha, warped_prev, alpha, 0.0)
                        final_out = (init_cv * (1 - mask_f) + blended * mask_f).astype(np.uint8)
                    except Exception as e:  # pragma: no cover
                        logger.warning("Flow/blend failed frame %d: %s", i, e)
                        final_out = current_fixed # Fallback

                out_path = os.path.join(out_dir, f"{i:06d}.png")
                cv2.imwrite(out_path, final_out)
                out_paths.append(out_path)

                if self.enable_full_debug:
                    os.makedirs(debug_frame_dir, exist_ok=True)
                    shutil.copy(out_path, os.path.join(debug_frame_dir, f"f{i:06d}_05_final.png"))

                prev_fixed = final_out.copy()
                prev_orig_gray = cv2.cvtColor(init_cv, cv2.COLOR_BGR2GRAY)

                if debug_upload_path_base and not self.enable_full_debug:
                    base = f"{i:06d}"
                    try:
                        upload_to_gcs(mask_frame_paths[i], f"{debug_upload_path_base}/{base}_mask.png")
                        upload_to_gcs(depth_map_paths[i], f"{debug_upload_path_base}/{base}_depth.png")
                        upload_to_gcs(out_path, f"{debug_upload_path_base}/{base}_fix.png")
                        vis_path = os.path.join(
                            FRAME_DIAGNOSTICS_DIR, f"frame_{i:06d}", f"f{i:06d}_04_3dvis.png"
                        )
                        if os.path.exists(vis_path):
                            upload_to_gcs(vis_path, f"{debug_upload_path_base}/{base}_3dvis.png")
                        else:
                            logger.debug("No 3D vis for upload at %s", vis_path)
                    except Exception as e:  # pragma: no cover
                        logger.warning("Debug upload failed @ frame %d: %s", i, e)

            except Exception as e:  # pragma: no cover
                logger.exception("Inpainting failed @ frame %d: %s", i, e)
                out_path = os.path.join(out_dir, f"{i:06d}.png")
                shutil.copy(gen_frame_paths[i], out_path)
                out_paths.append(out_path)
                prev_fixed = cv2.imread(gen_frame_paths[i])
                prev_orig_gray = (
                    cv2.cvtColor(prev_fixed, cv2.COLOR_BGR2GRAY)
                    if prev_fixed is not None
                    else None
                )

        logger.info("Inpainting OK. Processed %d frames.", len(out_paths))
        return out_paths, aligned_data_list

    # ---------- QA ----------
    def run_qa(
        self,
        target_aligned_data: List[List[Dict[str, Any]]],
        fixed_frame_paths: List[str],
        min_iou: float = 0.3,
        masks_dir_for_fixed: str = MASKS_DIR,
    ) -> Dict[str, Any]:
        logger.info("Running QA check...")
        logger.info("Extracting mesh from fixed frames...")
        fixed_mesh_data = self.get_hand_mesh_data(
            fixed_frame_paths, QA_MESH_DIR, masks_dir=masks_dir_for_fixed
        )
        min_len = min(len(target_aligned_data), len(fixed_mesh_data))
        if len(target_aligned_data) != len(fixed_mesh_data):
            logger.warning(
                "QA count mismatch T:%d, F:%d.", len(target_aligned_data), len(fixed_mesh_data)
            )
            target_aligned_data = target_aligned_data[:min_len]
            fixed_mesh_data = fixed_mesh_data[:min_len]

        errs: List[float] = []
        total_failed_matches = 0
        total_hands_compared = 0

        for frame_idx, (target_hands, fixed_hands) in enumerate(
            zip(target_aligned_data, fixed_mesh_data)
        ):
            logger.debug(
                "[QA F%06d] Comp %d target -> %d fixed.",
                frame_idx, len(target_hands), len(fixed_hands)
            )
            if not target_hands or not fixed_hands:
                continue

            target_bboxes = [t["bbox"] for t in target_hands]
            fixed_bboxes = [f["bbox"] for f in fixed_hands]
            cost_matrix = np.ones(
                (len(target_bboxes), len(fixed_bboxes)), dtype=np.float64
            )
            for i, tb in enumerate(target_bboxes):
                for j, fb in enumerate(fixed_bboxes):
                    cost_matrix[i, j] = 1.0 - self._calculate_iou(tb, fb)
            logger.debug("[QA F%06d] Cost Matrix:\n%s", frame_idx, cost_matrix)

            pairs = hungarian_or_greedy(cost_matrix)
            for i, j in pairs:
                if cost_matrix[i, j] > (1.0 - min_iou):
                    total_failed_matches += 1
                    continue
                try:
                    target_hand = target_hands[i]; fixed_hand = fixed_hands[j]
                    s, R, t = umeyama_align(
                        fixed_hand["pred_vertices"], target_hand["aligned_verts"]
                    )
                    qa_aligned_verts = (s * (fixed_hand["pred_vertices"] @ R.T)) + t
                    tv = target_hand["aligned_verts"]; qv = qa_aligned_verts
                    e = np.sqrt(np.sum((tv - qv) ** 2, axis=1))
                    mve_mm = float(np.mean(e) * 1000.0)
                    errs.append(mve_mm); total_hands_compared += 1
                    logger.debug(
                        "[QA F%06d] Match %d->%d MVE: %.2f mm", frame_idx, i, j, mve_mm
                    )
                except Exception as e:  # pragma: no cover
                    logger.error(
                        "QA MVE calc failed hand %d->%d frame %d: %s",
                        i, j, frame_idx, e
                    )
                    total_failed_matches += 1

        if not errs:
            logger.warning("QA failed: no valid hand pairs.")
            return {"pass": False, "reason": "No valid hand pairs."}

        avg_mve = float(np.mean(errs))
        qa_pass = avg_mve <= QA_MVE_THRESHOLD
        logger.log(
            logging.INFO if qa_pass else logging.WARNING,
            "QA %s Avg MVE: %.2f mm",
            "PASSED" if qa_pass else "FAILED",
            avg_mve,
        )
        return {
            "pass": qa_pass,
            "avg_mean_vertex_error_mm": avg_mve,
            "compared_hands": total_hands_compared,
            "total_frames": len(target_aligned_data),
            "failed_matches": total_failed_matches,
        }


# ===============================
# Main
# ===============================
def main() -> None:
    try:
        DRIVING_VIDEO_GCS = os.environ["DRIVING_VIDEO"]
        GENERATED_VIDEO_GCS = os.environ["GENERATED_VIDEO"]

        bucket_name, _ = parse_gcs_path(DRIVING_VIDEO_GCS)
        job_id = f"hand-fix-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        GCS_OUTPUT_PATH = f"gs://{bucket_name}/jobs/{job_id}/hand-fix-output"
        GCS_MODEL_PATH = f"gs://{bucket_name}/models"

        logger.info("Starting Hand-Fix Job: %s", job_id)
        logger.info("  Driving Video: %s", DRIVING_VIDEO_GCS)
        logger.info("  Generated Video: %s", GENERATED_VIDEO_GCS)
        logger.info("  GCS Model Path: %s", GCS_MODEL_PATH)
        logger.info("  Output Path: %s", GCS_OUTPUT_PATH)
        logger.info("  QA Threshold: %.2f mm", QA_MVE_THRESHOLD)
        logger.info("  Max Retries: %d", MAX_RETRY_ATTEMPTS)
        logger.info("  Mask Dilation: %d px", MASK_DILATION_PIXELS)
        logger.info("  Mask Feather: %.2f sigma", MASK_FEATHER_SIGMA)
        logger.info("  Min Hand Area: %d px", MIN_HAND_AREA_PIXELS)
        logger.info("  Temporal Smooth α: %.3f", TEMPORAL_SMOOTHING_ALPHA)
        logger.info("  SDXL Steps: %d", SDXL_NUM_INFERENCE_STEPS)
        logger.info("  SDXL CFG Scale: %.2f", SDXL_GUIDANCE_SCALE)
        logger.info("  SDXL CNet Scale: %.2f", SDXL_CONTROLNET_SCALE)
        logger.info("  SDXL Strength: %.2f", SDXL_STRENGTH)
        logger.info("  SDXL CPU Offload: %s", SDXL_CPU_OFFLOAD_ENABLED)
        logger.info("  SDXL Seed: %d", SDXL_SEED)
        logger.info("  YOLO Conf (SAM): %.2f, YOLO Conf (CROP): %.2f", YOLO_CONF_SAM, YOLO_CONF_CROP)
        logger.info("  Full Debug: %s", ENABLE_FULL_DEBUG)
        logger.info("  Deterministic: %s", DETERMINISTIC)

        # Download model artifacts
        logger.info("Downloading models...")
        
        # --- Define GCS paths ---
        graphormer_gcs_root = f"{GCS_MODEL_PATH}/graphormer"
        mano_gcs_root = f"{GCS_MODEL_PATH}/mano"
        hrnet_gcs_root = f"{GCS_MODEL_PATH}/hrnet"
        sam2_gcs_root = f"{GCS_MODEL_PATH}/sam2"
        yolo_gcs_root = f"{GCS_MODEL_PATH}/yolo"
        
        graphormer_checkpoint_gcs = f"{graphormer_gcs_root}/{MESHGRAPHORMER_CHECKPOINT_NAME}"
        yolo_checkpoint_gcs = f"{yolo_gcs_root}/{YOLO_CHECKPOINT}"

        # --- [MODIFIED] Ensure local directories exist ---
        os.makedirs(os.path.dirname(MESHGRAPHORMER_CHECKPOINT), exist_ok=True)
        os.makedirs(HRNET_LOCAL_DIR, exist_ok=True)
        os.makedirs(MESHGRAPHORMER_DATA_DIR, exist_ok=True)
        # Create the new root dir for all SAM2 models
        os.makedirs(SAM2_MODEL_ROOT_LOCAL, exist_ok=True) 


        downloads_ok = all(
            [
                # Download single files
                download_from_gcs(graphormer_checkpoint_gcs, MESHGRAPHORMER_CHECKPOINT),
                download_from_gcs(yolo_checkpoint_gcs, YOLO_CHECKPOINT_PATH),
                
                # Download folders to their respective target dirs
                download_gcs_folder(mano_gcs_root, MESHGRAPHORMER_DATA_DIR),
                download_gcs_folder(graphormer_gcs_root, MESHGRAPHORMER_DATA_DIR),
                download_gcs_folder(hrnet_gcs_root, HRNET_LOCAL_DIR),
                
                # --- [MODIFIED] ---
                # Download the entire SAM2 folder at once
                download_gcs_folder(sam2_gcs_root, SAM2_MODEL_ROOT_LOCAL),
            ]
        )
        if not downloads_ok:
            raise RuntimeError("Failed to download one or more models.")

        # Clean & (re)create local dirs
        logger.info("Cleaning local dirs...")
        all_dirs = [
            TEMP_DIR, DRIVING_FRAMES_DIR, DRIVING_MESH_DIR, TARGET_DEPTH_DIR,
            GEN_FRAMES_DIR, GEN_MESH_DIR, MASKS_DIR, RETRY_MASKS_DIR,
            FIXED_FRAMES_DIR, QA_MESH_DIR, DIAGNOSTICS_DIR, FRAME_DIAGNOSTICS_DIR
        ]
        for d in all_dirs:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        download_from_gcs(DRIVING_VIDEO_GCS, DRIVING_VIDEO_LOCAL)
        download_from_gcs(GENERATED_VIDEO_GCS, GEN_VIDEO_LOCAL)

        gen_w, gen_h, gen_fps = get_video_metadata(GEN_VIDEO_LOCAL)
        driv_w, driv_h, driv_fps = get_video_metadata(DRIVING_VIDEO_LOCAL)
        logger.info("  Gen Video: %dx%d @ %.2f FPS", gen_w, gen_h, gen_fps)
        logger.info("  Driving Vid: %dx%d @ %.2f FPS", driv_w, driv_h, driv_fps)

        # Build pipeline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = HandFixPipeline(device)
        pipeline.load_models()

        # --- Pipeline Execution ---
        logger.info("Extracting driving frames & meshes...")
        driving_frame_paths = extract_frames(DRIVING_VIDEO_LOCAL, DRIVING_FRAMES_DIR)
        driving_mesh_data = pipeline.get_hand_mesh_data(
            driving_frame_paths, DRIVING_MESH_DIR, masks_dir=None
        )

        logger.info("Extracting generated frames & meshes...")
        gen_frame_paths = extract_frames(GEN_VIDEO_LOCAL, GEN_FRAMES_DIR)

        logger.info("Generating hand masks...")
        masks_ok = pipeline.get_hand_masks_sam(
            GEN_VIDEO_LOCAL, gen_frame_paths, MASKS_DIR # Pass video path
        )
        if not masks_ok:
            raise RuntimeError("Failed to generate hand masks.")

        gen_mesh_data = pipeline.get_hand_mesh_data(
            gen_frame_paths, GEN_MESH_DIR, masks_dir=MASKS_DIR
        )

        if sum(len(f) for f in driving_mesh_data) == 0:
            raise RuntimeError("No hands found in driving video.")
        if sum(len(f) for f in gen_mesh_data) == 0:
            raise RuntimeError("No hands found in generated video.")

        mask_frame_paths = sorted(glob.glob(os.path.join(MASKS_DIR, "*.png")))
        min_len = min(
            len(mask_frame_paths), len(gen_frame_paths),
            len(driving_mesh_data), len(gen_mesh_data)
        )
        if len(mask_frame_paths) != len(gen_frame_paths):
            logger.warning("Frame/Mask mismatch! Processing %d frames.", min_len)

        # Trim to common length
        mask_frame_paths = mask_frame_paths[:min_len]
        gen_frame_paths = gen_frame_paths[:min_len]
        driving_mesh_data = driving_mesh_data[:min_len]
        gen_mesh_data = gen_mesh_data[:min_len]
        driving_frame_paths = driving_frame_paths[:min_len]

        qa_results: Dict[str, Any] = {}
        final_video_path = ""
        target_aligned_data: List[List[Dict[str, Any]]] = []

        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            logger.info("--- Fix Attempt %d/%d ---", attempt, MAX_RETRY_ATTEMPTS)
            current_mask_paths = mask_frame_paths
            current_fixed_dir = os.path.join(FIXED_FRAMES_DIR, f"v{attempt}")
            os.makedirs(current_fixed_dir, exist_ok=True)

            if attempt > 1:
                logger.info("QA failed. Dilating masks...")
                current_mask_dir = os.path.join(RETRY_MASKS_DIR, f"v{attempt}")
                os.makedirs(current_mask_dir, exist_ok=True)
                temp_new_masks: List[str] = []
                for mpath in mask_frame_paths:
                    new_path = os.path.join(current_mask_dir, os.path.basename(mpath))
                    try:
                        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                        if m is None:
                            m = np.zeros((gen_h, gen_w), dtype=np.uint8)
                        elif MASK_DILATION_KERNEL is not None:
                            m = cv2.dilate(m, MASK_DILATION_KERNEL, iterations=1)
                        cv2.imwrite(new_path, m)
                    except Exception as e:  # pragma: no cover
                        logger.error("Dilate mask failed %s: %s", mpath, e)
                        cv2.imwrite(new_path, np.zeros((gen_h, gen_w), dtype=np.uint8))
                    temp_new_masks.append(new_path)
                current_mask_paths = sorted(temp_new_masks)

                if len(current_mask_paths) != len(gen_frame_paths):
                    raise RuntimeError("Mask count mismatch after dilation.")

            GCS_DEBUG_PATH = f"{GCS_OUTPUT_PATH}/test/attempt_{attempt}"
            if not ENABLE_FULL_DEBUG:
                logger.info("Per-frame debug uploads to: %s", GCS_DEBUG_PATH)

            fixed_frame_paths, target_aligned_data = pipeline.run_guided_inpainting(
                gen_frame_paths=gen_frame_paths,
                mask_frame_paths=current_mask_paths,
                driving_mesh_data=driving_mesh_data,
                gen_mesh_data=gen_mesh_data,
                out_dir=current_fixed_dir,
                debug_upload_path_base=GCS_DEBUG_PATH if not ENABLE_FULL_DEBUG else None,
            )

            qa_results = pipeline.run_qa(
                target_aligned_data, fixed_frame_paths, masks_dir_for_fixed=MASKS_DIR
            )

            final_video_path = os.path.join(TEMP_DIR, f"fixed_video_v{attempt}.mp4")
            frames_to_video(current_fixed_dir, final_video_path, gen_fps)

            if qa_results.get("pass"):
                logger.info("Attempt %d PASSED QA.", attempt)
                break
            else:
                logger.warning("Attempt %d FAILED QA. %s", attempt, qa_results)
                if attempt == MAX_RETRY_ATTEMPTS:
                    logger.error("Max retries reached. Job failed QA.")

        # --- Upload final results ---
        diagnostics = {
            "job_id": job_id,
            "driving_video": DRIVING_VIDEO_GCS,
            "generated_video": GENERATED_VIDEO_GCS,
            "qa_status": "PASS" if qa_results.get("pass") else "FAIL",
            "qa_details": qa_results,
            "pipeline_type": (
                "SDXL_ControlNet_Inpaint (Multi-hand, Umeyama + zbuf depth + mask crop + smoothing)"
            ),
        }
        diag_file = os.path.join(DIAGNOSTICS_DIR, "diagnostics.json")
        with open(diag_file, "w") as f:
            json.dump(diagnostics, f, indent=4, default=str)
        upload_to_gcs(DIAGNOSTICS_DIR, f"{GCS_OUTPUT_PATH}/diagnostics")

        if qa_results.get("pass"):
            upload_to_gcs(final_video_path, f"{GCS_OUTPUT_PATH}/fixed_video_PASSED.mp4")
            logger.info("Job completed successfully.")
        else:
            if os.path.exists(final_video_path):
                upload_to_gcs(final_video_path, f"{GCS_OUTPUT_PATH}/fixed_video_FAILED.mp4")

            # Upload detailed diagnostics on failure
            upload_to_gcs(MASKS_DIR, f"{GCS_OUTPUT_PATH}/diagnostics/masks_v1")
            upload_to_gcs(TARGET_DEPTH_DIR, f"{GCS_OUTPUT_PATH}/diagnostics/target_depth_maps_scaled")
            upload_to_gcs(DRIVING_MESH_DIR, f"{GCS_OUTPUT_PATH}/diagnostics/driving_mesh_data")
            upload_to_gcs(GEN_MESH_DIR, f"{GCS_OUTPUT_PATH}/diagnostics/gen_mesh_data")
            upload_to_gcs(QA_MESH_DIR, f"{GCS_OUTPUT_PATH}/diagnostics/qa_mesh_data")
            if os.path.exists(RETRY_MASKS_DIR):
                upload_to_gcs(RETRY_MASKS_DIR, f"{GCS_OUTPUT_PATH}/diagnostics/masks_v2_dilated")
            logger.error("Job completed but FAILED QA. Details: %s", qa_results)

        if ENABLE_FULL_DEBUG:
            logger.info("Uploading per-frame diagnostics...")
            upload_to_gcs(FRAME_DIAGNOSTICS_DIR, f"{GCS_OUTPUT_PATH}/frame_diagnostics")

    except Exception as e:  # pragma: no cover
        logger.exception("Unhandled exception in main: %s", e)
        try:
            os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)
            diag_file = os.path.join(DIAGNOSTICS_DIR, "diagnostics_CRASH.json")
            with open(diag_file, "w") as f:
                json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)
            if "GCS_OUTPUT_PATH" in locals():
                upload_to_gcs(DIAGNOSTICS_DIR, f"{GCS_OUTPUT_PATH}/diagnostics")
        except Exception as e_diag:  # pragma: no cover
            logger.error("Could not upload crash diagnostics: %s", e_diag)
        sys.exit(1)
    finally:
        logger.info("Cleaning up local directory: %s", TEMP_DIR)
        # If you want to clean up, uncomment below:
        # shutil.rmtree(TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
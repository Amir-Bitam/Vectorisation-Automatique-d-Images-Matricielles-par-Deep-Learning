import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pydiffvg
import torch
import torchvision.utils as vutils
import yaml
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiment import VAEXperiment
from models import vae_models


REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DOMAIN = "emoji"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_config(config: Dict, repo_root: Path = REPO_ROOT) -> Dict:
    prepared = copy.deepcopy(config)
    data_path = Path(prepared["exp_params"]["data_path"])
    if not data_path.is_absolute():
        data_path = (repo_root / data_path).resolve()
    prepared["exp_params"]["data_path"] = str(data_path)
    if os.name == "nt":
        prepared["exp_params"].setdefault("num_workers", 0)
    return prepared


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_load_checkpoint(checkpoint_path: Path) -> Dict:
    return torch.load(checkpoint_path, map_location="cpu")


def build_experiment(
    config: Dict,
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[VAEXperiment, Dict]:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the validated Windows Im2Vec workflow.")

    pydiffvg.set_use_gpu(True)
    model = vae_models[config["model_params"]["name"]](
        imsize=config["exp_params"]["img_size"],
        **config["model_params"],
    )
    experiment = VAEXperiment(model, config["exp_params"])
    load_info = {
        "checkpoint_epoch": None,
        "global_step": None,
        "strict_load": None,
        "missing_keys": [],
        "unexpected_keys": [],
        "checkpoint_pytorch_lightning_version": None,
    }

    if checkpoint_path is not None:
        checkpoint = torch_load_checkpoint(checkpoint_path)
        result = experiment.load_state_dict(checkpoint["state_dict"], strict=True)
        load_info.update(
            {
                "checkpoint_epoch": checkpoint.get("epoch"),
                "global_step": checkpoint.get("global_step"),
                "strict_load": True,
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
                "checkpoint_pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
            }
        )

    experiment.to(device)
    experiment.eval()
    experiment.freeze()
    return experiment, load_info


def preprocessing_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )


def preprocess_image(image_path: Path, img_size: int) -> Tuple[torch.Tensor, Dict]:
    source = Image.open(image_path)
    original_mode = source.mode
    original_size = source.size
    rgb = source.convert("RGB")
    tensor = preprocessing_transform(img_size)(rgb).unsqueeze(0)
    return tensor, {
        "original_mode": original_mode,
        "original_size": list(original_size),
        "model_input_size": [img_size, img_size],
        "alpha_handling": "PIL convert('RGB'), matching the repository dataset loader",
        "normalization": "ToTensor only, values in [0, 1]",
    }


def load_dataset_batch(config: Dict, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    batch_size = batch_size or int(config["exp_params"]["val_batch_size"])
    transform = preprocessing_transform(int(config["exp_params"]["img_size"]))
    dataset = datasets.ImageFolder(config["exp_params"]["data_path"], transform=transform)
    if len(dataset) < batch_size:
        raise RuntimeError(f"Dataset contains {len(dataset)} images, fewer than requested batch size {batch_size}.")
    generator = torch.Generator()
    generator.manual_seed(int(config["logging_params"]["manual_seed"]))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(config["exp_params"].get("val_shuffle", True)),
        drop_last=True,
        num_workers=int(config["exp_params"].get("num_workers", 0)),
        generator=generator,
    )
    images, labels = next(iter(loader))
    first_paths = [dataset.samples[index][0] for index in range(min(batch_size, len(dataset.samples)))]
    return images, labels, first_paths


def reset_curve_count(model: torch.nn.Module, paths: int) -> None:
    if hasattr(model, "redo_features"):
        model.redo_features(int(paths))


def encode_to_latent(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    mu, log_var = model.encode(images)
    return model.reparameterize(mu, log_var)


def reconstruct_from_latent(model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "decode_and_composite"):
        return model.decode_and_composite(z, verbose=False)
    return model.raster(model.decode(z), verbose=False)[:, :3]


def layer_points_from_latent(model: torch.nn.Module, z: torch.Tensor) -> List[Tuple[torch.Tensor, List[float]]]:
    if hasattr(model, "decode_and_composite") and hasattr(model, "colors"):
        n_layers = len(model.colors)
        z_rnn_input = z[None, :, :].repeat(n_layers, 1, 1)
        outputs, _ = model.rnn(z_rnn_input)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs[:, :, : model.latent_dim] + outputs[:, :, model.latent_dim :]

        layers = []
        for layer_index in range(n_layers):
            shape_output = model.divide_shape(outputs[:, layer_index, :])
            shape_latent = model.final_shape_latent(shape_output)
            layers.append((model.decode(shape_latent), model.colors[layer_index]))
        return layers

    return [(model.decode(z), [0.0, 0.0, 0.0, 1.0])]


def save_latent_svg(model: torch.nn.Module, z: torch.Tensor, output_path: Path, sample_index: int = 0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = layer_points_from_latent(model, z)
    shapes = []
    shape_groups = []
    render_size = int(model.imsize)
    curve_count = int(model.curves)
    num_ctrl_pts = torch.zeros(curve_count, dtype=torch.int32) + 2

    for layer_points, color in layers:
        points = (layer_points[sample_index].detach().cpu().contiguous() * render_size)
        path = pydiffvg.Path(
            num_control_points=num_ctrl_pts,
            points=points,
            is_closed=True,
        )
        shapes.append(path)
        color_tensor = torch.tensor(color, dtype=torch.float32)
        shape_groups.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1], dtype=torch.int32),
                fill_color=color_tensor,
                stroke_color=color_tensor,
            )
        )

    pydiffvg.save_svg(str(output_path), render_size, render_size, shapes, shape_groups)


def save_tensor_image(tensor: torch.Tensor, output_path: Path, nrow: int = 1) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(tensor.detach().cpu().clamp(0, 1), str(output_path), normalize=False, nrow=nrow)


def render_svg_with_diffvg(svg_path: Path, output_png: Path) -> Dict:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(str(svg_path))
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    img = pydiffvg.RenderFunction.apply(
        canvas_width,
        canvas_height,
        3,
        3,
        102,
        None,
        *scene_args,
    )
    pydiffvg.imwrite(img.detach().cpu(), str(output_png), gamma=1.0)
    return {"width": int(canvas_width), "height": int(canvas_height)}


def _local_xml_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def validate_svg(svg_path: Path, rerasterized_png: Optional[Path] = None) -> Dict:
    if not svg_path.exists():
        raise FileNotFoundError(svg_path)
    if svg_path.stat().st_size <= 0:
        raise RuntimeError(f"SVG file is empty: {svg_path}")

    text = svg_path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"\b(?:nan|inf)\b", text, flags=re.IGNORECASE):
        raise RuntimeError(f"SVG contains NaN or Inf: {svg_path}")

    root = ET.fromstring(text)
    if _local_xml_name(root.tag).lower() != "svg":
        raise RuntimeError(f"Root element is not <svg>: {svg_path}")

    path_elements = []
    primitive_count = 0
    group_count = 0
    fill_values = []
    for element in root.iter():
        name = _local_xml_name(element.tag).lower()
        if name == "path":
            path_elements.append(element)
        elif name in {"rect", "circle", "ellipse", "line", "polyline", "polygon"}:
            primitive_count += 1
        elif name == "g":
            group_count += 1
        if "fill" in element.attrib:
            fill_values.append(element.attrib["fill"])
        style = element.attrib.get("style", "")
        match = re.search(r"(?:^|;)\s*fill\s*:\s*([^;]+)", style)
        if match:
            fill_values.append(match.group(1))

    for element in path_elements:
        data = element.attrib.get("d", "").strip()
        if not data:
            raise RuntimeError(f"SVG path has empty data: {svg_path}")
        for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", data):
            if not math.isfinite(float(value)):
                raise RuntimeError(f"SVG path contains a non-finite coordinate: {svg_path}")

    raster_info = {}
    if rerasterized_png is not None:
        raster_info = render_svg_with_diffvg(svg_path, rerasterized_png)
        image = Image.open(rerasterized_png).convert("RGBA")
        array = np.asarray(image)
        flat = array.reshape(-1, array.shape[-1])
        raster_info.update(
            {
                "rerasterized_path": str(rerasterized_png.resolve()),
                "transparent": bool(array[..., 3].max() == 0),
                "constant": bool(np.all(flat == flat[0])),
                "black": bool(array[..., :3].max() == 0),
                "white": bool(array[..., :3].min() == 255),
            }
        )
        if raster_info["transparent"] or raster_info["constant"]:
            raise RuntimeError(f"Rerasterized SVG is transparent or constant: {svg_path}")

    return {
        "svg_path": str(svg_path.resolve()),
        "size_bytes": svg_path.stat().st_size,
        "root": "svg",
        "path_count": len(path_elements),
        "primitive_count": primitive_count,
        "fill_count": len(fill_values),
        "group_count": group_count,
        **raster_info,
    }


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")


def make_side_by_side(input_tensor: torch.Tensor, reconstruction_png: Path, rerasterized_png: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [
        tensor_to_pil(input_tensor[0]),
        Image.open(reconstruction_png).convert("RGB").resize((input_tensor.shape[-1], input_tensor.shape[-2])),
        Image.open(rerasterized_png).convert("RGB").resize((input_tensor.shape[-1], input_tensor.shape[-2])),
    ]
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    canvas.save(output_path)


def reconstruction_metrics(input_tensor: torch.Tensor, reconstruction_tensor: torch.Tensor) -> Dict:
    input_arr = input_tensor[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    recon_arr = reconstruction_tensor[0, :3].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    mse = float(np.mean((input_arr - recon_arr) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
    ssim = float(structural_similarity(input_arr, recon_arr, channel_axis=2, data_range=1.0))
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def reconstruct_tensor(
    experiment: VAEXperiment,
    input_tensor: torch.Tensor,
    output_dir: Path,
    stem: str,
    config: Dict,
    save_svg: bool = True,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(experiment.parameters()).device
    reset_curve_count(experiment.model, int(config["model_params"]["paths"]))
    input_on_device = input_tensor.to(device)
    with torch.no_grad():
        z = encode_to_latent(experiment.model, input_on_device)
        reconstruction = reconstruct_from_latent(experiment.model, z)

    input_png = output_dir / f"{stem}_input.png"
    reconstruction_png = output_dir / f"{stem}_reconstruction.png"
    save_tensor_image(input_tensor, input_png)
    save_tensor_image(reconstruction[:, :3], reconstruction_png)

    result = {
        "input_png": str(input_png.resolve()),
        "reconstruction_png": str(reconstruction_png.resolve()),
        "metrics": reconstruction_metrics(input_tensor, reconstruction),
    }
    if save_svg:
        svg_path = output_dir / f"{stem}.svg"
        rerasterized_png = output_dir / f"{stem}_rerasterized.png"
        comparison_png = output_dir / f"{stem}_comparison.png"
        save_latent_svg(experiment.model, z, svg_path, sample_index=0)
        validation = validate_svg(svg_path, rerasterized_png)
        make_side_by_side(input_tensor, reconstruction_png, rerasterized_png, comparison_png)
        result.update(
            {
                "svg_path": str(svg_path.resolve()),
                "rerasterized_png": str(rerasterized_png.resolve()),
                "comparison_png": str(comparison_png.resolve()),
                "svg_validation": validation,
            }
        )
    return result


def package_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def write_environment_report(
    report_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    repository_commit: str,
    pydiffvg_path: str,
    successful_official_evaluation: bool,
    custom_image_supported: bool,
    compatibility_patches: Iterable[str],
) -> Dict:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cuda_available = torch.cuda.is_available()
    report = {
        "method": "Im2Vec",
        "repository_commit": repository_commit,
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_domain": CHECKPOINT_DOMAIN,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "torchvision_version": package_version("torchvision"),
        "pytorch_lightning_version": package_version("pytorch_lightning"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(cuda_available),
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "",
        "gpu_compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if cuda_available else "",
        "pydiffvg_path": pydiffvg_path,
        "native_windows": True,
        "successful_official_evaluation": bool(successful_official_evaluation),
        "custom_image_supported": bool(custom_image_supported),
        "compatibility_patches": list(compatibility_patches),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

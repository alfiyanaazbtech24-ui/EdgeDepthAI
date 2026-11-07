"""
EdgeDepth AI Flask API
Place this file at the project root as app.py.

Expected project modules:
  backend.preprocessing, backend.reconstruction, backend.depth_refinement
  backend.edge_detection
  classical.bilateral, classical.guided_filter, classical.hole_filling
  classical.boundary_refinement
"""
import base64
import io
import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from backend.preprocessing import load_rgb, load_depth, validate_rgb_depth
from backend.depth_refinement import refine_depth_map
from backend.reconstruction import reconstruct

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-in-production")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

ALLOWED_RGB = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ALLOWED_DEPTH = ALLOWED_RGB | {".npy", ".pfm"}


def _save_upload(file_storage, allowed):
    suffix = Path(secure_filename(file_storage.filename or "")).suffix.lower()
    if suffix not in allowed:
        raise ValueError(f"Unsupported file extension: {suffix or '(none)'}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file_storage.save(tmp)
    tmp.close()
    return Path(tmp.name)


def _depth_preview(depth, valid=None):
    depth = np.asarray(depth, np.float32)
    mask = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        mask &= np.asarray(valid, bool)
    vis = np.zeros(depth.shape, np.uint8)
    if mask.any():
        lo, hi = np.percentile(depth[mask], [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        vis[mask] = np.clip((depth[mask] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def _png_data_url(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode output preview.")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def _metric_values(prediction, target):
    valid = (
        np.isfinite(prediction) & np.isfinite(target)
        & (prediction > 0) & (target > 0)
    )
    if not valid.any():
        return {"mae": None, "rmse": None, "psnr": None, "ssim": None}
    p = prediction[valid].astype(np.float64)
    t = target[valid].astype(np.float64)
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    peak = float(np.max(t) - np.min(t))
    psnr = float(20 * np.log10(max(peak, 1e-12) / max(rmse, 1e-12)))
    # SSIM on valid pixels only is not well-defined near holes. Use masked
    # global luminance/contrast approximation, returning a stable scalar.
    mu_p, mu_t = float(p.mean()), float(t.mean())
    var_p, var_t = float(p.var()), float(t.var())
    cov = float(np.mean((p - mu_p) * (t - mu_t)))
    c1 = (0.01 * max(peak, 1e-6)) ** 2
    c2 = (0.03 * max(peak, 1e-6)) ** 2
    ssim = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / (
        (mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2)
    )
    return {
        "mae": round(mae, 6), "rmse": round(rmse, 6),
        "psnr": round(psnr, 4), "ssim": round(float(np.clip(ssim, -1, 1)), 6)
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="running", application="EdgeDepthAI")


@app.post("/reconstruct")
def reconstruct_api():
    temp_paths = []
    started = time.perf_counter()
    try:
        rgb_file = request.files.get("rgb")
        depth_file = request.files.get("depth")
        gt_file = request.files.get("ground_truth")
        if not rgb_file or not depth_file:
            return jsonify(error="Upload both an RGB image and a depth map."), 400

        rgb_path = _save_upload(rgb_file, ALLOWED_RGB)
        depth_path = _save_upload(depth_file, ALLOWED_DEPTH)
        temp_paths += [rgb_path, depth_path]
        rgb = load_rgb(rgb_path)
        depth, valid = load_depth(depth_path)
        validate_rgb_depth(rgb, depth)

        scale = int(request.form.get("scale", 2))
        if scale not in (2, 4, 8):
            raise ValueError("Scale must be 2, 4, or 8.")
        method = request.form.get("method", "hybrid").lower()
        allowed_methods = {"nearest", "bilinear", "bicubic", "joint_bilateral", "guided", "hybrid", "deep"}
        if method not in allowed_methods:
            raise ValueError(f"Unsupported method: {method}")

        # Deep model inference is intentionally not faked. It needs a separately
        # exported/inference-ready checkpoint and matching input preprocessing.
        if method == "deep":
            return jsonify(error=(
                "Deep inference is not wired into this API yet. "
                "Choose Hybrid or a classical method; the trained checkpoint "
                "needs a dedicated inference adapter."
            )), 501

        base_method = method if method in {"nearest", "bilinear", "bicubic"} else "bilinear"
        base = reconstruct(depth, scale=scale, method=base_method, valid_mask=valid)
        up_depth = base["depth"]
        up_valid = base["confidence"] > 1e-5

        # Upscale RGB to match the reconstructed depth resolution.
        out_h, out_w = up_depth.shape
        rgb_up = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        # The selected optional filters run on the high-resolution result.
        use_rgb = request.form.get("rgb_guidance", "true").lower() in {"true", "1", "yes"}
        use_boundary = request.form.get("boundary_refinement", "true").lower() in {"true", "1", "yes"}
        noise = request.form.get("noise_reduction", "none").lower()
        hole = request.form.get("hole_filling", "none").lower()
        edge_enabled = request.form.get("edge_preservation", "enabled").lower() != "disabled"

        refined = up_depth.copy()
        if hole in {"nearest", "edge_aware"} and (~up_valid).any() and up_valid.any():
            from classical.hole_filling import fill_depth_holes
            refined, filled = fill_depth_holes(
                refined, valid_mask=up_valid,
                max_distance=12 if hole == "edge_aware" else None,
                return_mask=True
            )
            up_valid = up_valid | filled

        if noise == "median":
            safe = refined.copy()
            if up_valid.any():
                safe[~up_valid] = float(np.median(safe[up_valid]))
            refined = cv2.medianBlur(safe.astype(np.float32), 3)
            refined[~up_valid] = 0
        elif noise == "bilateral" or method == "joint_bilateral":
            from classical.bilateral import bilateral_refine
            refined = bilateral_refine(refined, valid_mask=up_valid, diameter=7,
                                       sigma_color=max(float(np.std(refined[up_valid])) * 0.08, 1e-4),
                                       sigma_space=7)

        if method in {"guided", "hybrid"} and use_rgb:
            from classical.guided_filter import guided_filter
            refined = guided_filter(rgb_up, refined, radius=8, eps=0.01, valid_mask=up_valid)

        if method == "hybrid" and edge_enabled and use_boundary:
            from classical.boundary_refinement import refine_boundaries
            refined = refine_boundaries(
                rgb_up, refined, valid_mask=up_valid,
                radius=8, eps=0.01, edge_preservation=0.85
            )

        refined = np.nan_to_num(refined, nan=0, posinf=0, neginf=0).astype(np.float32)
        refined[~up_valid] = 0

        edge_gray = cv2.cvtColor(rgb_up, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(edge_gray, 60, 140)
        depth_vis = _depth_preview(refined, up_valid)
        original_vis = _depth_preview(depth, valid)
        boundary_vis = cv2.applyColorMap(edges, cv2.COLORMAP_INFERNO)

        metrics = {"mae": None, "rmse": None, "psnr": None, "ssim": None}
        if gt_file and gt_file.filename:
            gt_path = _save_upload(gt_file, ALLOWED_DEPTH)
            temp_paths.append(gt_path)
            gt, gt_valid = load_depth(gt_path)
            if gt.shape != refined.shape:
                gt = cv2.resize(gt, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                gt_valid = cv2.resize(gt_valid.astype(np.uint8), (out_w, out_h),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            metric_mask = gt_valid & up_valid
            metrics = _metric_values(
                np.where(metric_mask, refined, 0),
                np.where(metric_mask, gt, 0)
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return jsonify(
            original=_png_data_url(original_vis),
            refined=_png_data_url(depth_vis),
            edges=_png_data_url(edges),
            boundary=_png_data_url(boundary_vis),
            **metrics,
            processing_time=round(elapsed_ms, 2),
            output_resolution=f"{out_w} × {out_h}",
            memory_usage=None,
            scale=scale,
            method=method,
            valid_pixels=int(up_valid.sum())
        )
    except (ValueError, FileNotFoundError) as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        app.logger.exception("Reconstruction failed")
        return jsonify(error=f"Reconstruction failed: {exc}"), 500
    finally:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

# Copyright 2026 The LOCI Authors.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Camera geometry used by LOCI's camera conditioning.

Conventions
-----------
* Poses are camera-to-world matrices (``[..., 3, 4]`` or ``[..., 4, 4]``) in the
  OpenCV convention: +x right, +y down, +z forward. The first latent frame of a
  trajectory is the world origin (identity rotation, zero translation).
* Intrinsics follow the unified camera model (UCM) of UCPE, parameterised by the
  horizontal field of view ``x_fov`` (degrees) and the UCM ``xi`` (0 = pinhole).
* All geometry is computed in float32; it is only rounded to the model dtype
  where the attention layers consume it.

The UCM projection / "up + latitude" map and the per-ray frames follow the
formulation of UCPE (https://github.com/chengzhag/UCPE, MIT license).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def per_frame_param(value, batch_size: int, frames: int, device, dtype) -> torch.Tensor:
    """Broadcast a scalar / ``[B]`` / ``[T]`` / ``[B, T]`` camera parameter to ``[B, T]``."""
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.ndim == 0:
        return value.expand(batch_size, frames)
    if value.ndim == 1:
        if value.shape[0] == batch_size:
            return value[:, None].expand(batch_size, frames)
        if value.shape[0] == frames and batch_size == 1:
            return value[None].expand(batch_size, frames)
    if value.ndim == 2 and value.shape == (batch_size, frames):
        return value
    raise ValueError(f"expected scalar, [B] or [B, T] camera parameter, got {tuple(value.shape)}")


def as_c2w_4x4(pose: torch.Tensor, batch_size: int) -> torch.Tensor:
    """``[T,3,4]`` / ``[B,T,3,4]`` / ``[B,T,4,4]`` camera-to-world -> ``[B,T,4,4]``."""
    if pose.ndim == 3:
        pose = pose.unsqueeze(0)
    if pose.shape[0] == 1 and batch_size > 1:
        pose = pose.expand(batch_size, *pose.shape[1:])
    if pose.shape[0] != batch_size:
        raise ValueError(f"expected pose batch {batch_size}, got {pose.shape[0]}")
    if pose.shape[-2:] == (4, 4):
        return pose
    if pose.shape[-2:] != (3, 4):
        raise ValueError(f"expected [B,T,3,4] or [B,T,4,4] poses, got {tuple(pose.shape)}")
    c2w = torch.zeros(*pose.shape[:-2], 4, 4, device=pose.device, dtype=pose.dtype)
    c2w[..., :3, :4] = pose
    c2w[..., 3, 3] = 1.0
    return c2w


def ucm_focal(x_fov: torch.Tensor, xi: torch.Tensor, width: int, dtype) -> torch.Tensor:
    """Focal length (in grid units) of a UCM camera with horizontal FOV ``x_fov``."""
    theta = torch.deg2rad(0.5 * x_fov)
    eps = torch.finfo(dtype).eps
    denom = torch.sin(theta).clamp_min(eps)
    return (width * 0.5) * (torch.cos(theta) + xi) / denom


def ucm_ray_grid(x_fov, xi, height: int, width: int, device, dtype=torch.float32) -> torch.Tensor:
    """Unit camera-space ray directions ``[B, H, W, 3]`` of the token grid (UCM inverse projection)."""
    x_fov = torch.as_tensor(x_fov, device=device, dtype=dtype).reshape(-1)
    xi = torch.as_tensor(xi, device=device, dtype=dtype).reshape(-1)
    batch = max(x_fov.shape[0], xi.shape[0])
    x_fov = x_fov.expand(batch)
    xi = xi.expand(batch)

    fx = ucm_focal(x_fov, xi, width, dtype)
    fy = fx
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    mx = (x[None] - width * 0.5) / fx[:, None, None]
    my = (y[None] - height * 0.5) / fy[:, None, None]
    rho2 = mx * mx + my * my
    sqrt_term = 1.0 + (1.0 - xi[:, None, None] * xi[:, None, None]) * rho2
    invalid = sqrt_term < 0.0
    alpha = (xi[:, None, None] + sqrt_term.clamp_min(0.0).sqrt()) / (rho2 + 1.0)
    dirs = torch.stack([alpha * mx, alpha * my, alpha - xi[:, None, None]], dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return dirs.masked_fill(invalid[..., None], 0.0)


def ucm_project(X, Y, Z, x_fov, xi, height: int, width: int):
    fx = ucm_focal(x_fov, xi, width, X.dtype)
    fy = fx
    while fx.ndim < X.ndim:
        fx = fx[..., None]
        fy = fy[..., None]
        xi = xi[..., None]
    radius = torch.sqrt(X * X + Y * Y + Z * Z)
    alpha = Z + xi * radius
    du = fx * (X / alpha.clamp_min(torch.finfo(X.dtype).eps)) + width * 0.5
    dv = fy * (Y / alpha.clamp_min(torch.finfo(Y.dtype).eps)) + height * 0.5
    return du, dv


def up_latitude_maps(R: torch.Tensor, x_fov, xi, height: int, width: int, delta: float = 0.1):
    """Per-token absolute gravity cues: image-plane "up" direction (2) and ray latitude (1).

    ``R``: camera-to-world rotations ``[B, T, 3, 3]``. Returns ``(up [B,T,H,W,2], lat [B,T,H,W,1])``.
    """
    batch_size, frame_count, _, _ = R.shape
    device = R.device
    out_dtype = R.dtype
    R = R.float()
    x_fov = per_frame_param(x_fov, batch_size, frame_count, device, torch.float32)
    xi = per_frame_param(xi, batch_size, frame_count, device, torch.float32)

    flat = batch_size * frame_count
    d_cam = ucm_ray_grid(x_fov.reshape(flat), xi.reshape(flat), height, width, device,
                         torch.float32).reshape(batch_size, frame_count, height, width, 3)
    mask = d_cam.abs().sum(dim=-1, keepdim=True) == 0.0

    d_world = torch.einsum("btij,bthwj->bthwi", R, d_cam)
    d_world = d_world / d_world.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    xw, yw, zw = d_world[..., 0], d_world[..., 1], d_world[..., 2]
    lat_map = torch.atan2(-yw, torch.sqrt(xw * xw + zw * zw)).unsqueeze(-1)

    up_world = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)
    axis = torch.cross(d_world, up_world.view(1, 1, 1, 1, 3).expand_as(d_world), dim=-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    delta = torch.tensor(delta, device=device, dtype=torch.float32)
    cos_delta = torch.cos(delta)
    sin_delta = torch.sin(delta)
    v_rot = (
        d_world * cos_delta
        + torch.cross(axis, d_world, dim=-1) * sin_delta
        + axis * (axis * d_world).sum(dim=-1, keepdim=True) * (1.0 - cos_delta)
    )
    dirs_cam = torch.einsum("btij,bthwj->bthwi", R.transpose(-1, -2), v_rot)
    du, dv = ucm_project(dirs_cam[..., 0], dirs_cam[..., 1], dirs_cam[..., 2], x_fov, xi, height, width)
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    up_map = torch.stack((du - x, dv - y), dim=-1)
    up_map = up_map / up_map.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    up_map = up_map.masked_fill(mask, 0.0).to(out_dtype)
    lat_map = lat_map.masked_fill(mask, 0.0).to(out_dtype)
    return up_map, lat_map


def ray_frames(d_cam: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Per-ray world-to-ray SE(3) matrices ``[B, T, H, W, 4, 4]``.

    Each token gets a local frame whose z axis is its world-space viewing ray and whose
    x axis is orthogonal to the camera's y axis; its origin is the camera centre.
    """
    batch_size, height, width, _ = d_cam.shape
    frame_count = c2w.shape[1]
    d_cam = repeat(d_cam, "b h w c -> b t h w c", t=frame_count)
    r_cam = c2w[..., :3, :3]
    t_cam = c2w[..., :3, 3]
    d_world = torch.einsum("btij,bthwj->bthwi", r_cam, d_cam)

    cam_y = repeat(r_cam[..., :, 1], "b t c -> b t h w c", h=height, w=width)
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-6)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-6)

    r_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)
    r_w2l = r_l2w.transpose(-1, -2)
    t_world = repeat(t_cam, "b t c -> b t h w c", h=height, w=width)
    t_w2l = -torch.einsum("bthwij,bthwj->bthwi", r_w2l, t_world)

    mats = torch.zeros(batch_size, frame_count, height, width, 4, 4, device=d_cam.device, dtype=d_cam.dtype)
    mats[..., :3, :3] = r_w2l
    mats[..., :3, 3] = t_w2l
    mats[..., 3, 3] = 1.0
    invalid = torch.isnan(d_world).any(dim=-1)
    if invalid.any():
        mats[invalid] = torch.eye(4, device=d_cam.device, dtype=d_cam.dtype)
    return mats


def pinhole_intrinsics_from_fov(x_fov: torch.Tensor, height: int, width: int, image_scale: int) -> torch.Tensor:
    """Isotropic pinhole ``K`` ``[B, T, 3, 3]`` on a ``(width*scale) x (height*scale)`` image."""
    K = torch.zeros(*x_fov.shape, 3, 3, device=x_fov.device, dtype=torch.float32)
    K[..., 0, 0] = K[..., 1, 1] = (width * image_scale) / (2 * torch.tan(x_fov * (math.pi / 360)))
    K[..., 0, 2], K[..., 1, 2], K[..., 2, 2] = width * (image_scale // 2), height * (image_scale // 2), 1
    return K


def build_camera_inputs(pose, x_fov, xi, grid_shape, *, recurrent_translation_scale: float,
                        recurrent_image_scale: int, model_dtype) -> dict:
    """Everything the camera-aware layers need for one chunk.

    Returns a dict with
      ``ray_mats``   [B, T*H*W, 4, 4] fp32 per-ray frames (UCPE camera attention),
      ``abs_map``    [B, T*H*W, 3] up/latitude map in the model dtype (UCPE camera encoder),
      ``rec_mats``   [B, T*H*W, 4, 4] fp32 per-ray frames with translation divided by
                     ``recurrent_translation_scale`` (PRoPE on the recurrent state addresses),
      ``rec_K``      [B, T*H*W, 3, 3] fp32 pinhole intrinsics for the same PRoPE.
    """
    frames, height, width = grid_shape
    batch = pose.shape[0] if pose.ndim == 4 else 1
    c2w = as_c2w_4x4(pose, batch)
    fov_bt = per_frame_param(x_fov, batch, frames, c2w.device, torch.float32)
    xi_bt = per_frame_param(xi, batch, frames, c2w.device, torch.float32)
    d_cam = ucm_ray_grid(fov_bt[:, 0], xi_bt[:, 0], height, width, c2w.device, torch.float32)
    mats = ray_frames(d_cam.float(), c2w.float())
    rec = mats.reshape(batch, frames * height * width, 4, 4).clone()
    rec[..., :3, 3] /= recurrent_translation_scale
    K = pinhole_intrinsics_from_fov(fov_bt, height, width, recurrent_image_scale)
    up, lat = up_latitude_maps(c2w[..., :3, :3], fov_bt, xi_bt, height, width)
    abs_map = torch.cat([up, lat], dim=-1).reshape(batch, frames * height * width, -1)
    return {
        "ray_mats": rearrange(mats, "b t h w ... -> b (t h w) ..."),
        "abs_map": abs_map.to(model_dtype),
        "rec_mats": rec,
        "rec_K": K.repeat_interleave(height * width, dim=1),
    }

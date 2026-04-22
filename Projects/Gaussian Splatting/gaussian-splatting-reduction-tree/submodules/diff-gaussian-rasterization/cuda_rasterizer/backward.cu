/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * --------------------------------------------------------------------
 * Reduction-tree / warp-per-pixel forward-compatible BACKWARD.cu
 * - Keeps preprocessing backward identical to original.
 * - Replaces BACKWARD::render with a warp-per-pixel kernel launch that
 *   matches your forward mapping (block = 256x1, grid = tiles).
 * - Backward per-pixel recurrence is inherently sequential, so lane0
 *   performs the recurrence; other lanes cooperate on chunk prefetch.
 * --------------------------------------------------------------------
 */

#include "backward.h"
#include "auxiliary.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ int bwd_debug_once = 0;
__device__ int g_bwd_print_once = 0;
__device__ int g_dbg_print_count = 0;
__device__ int g_bwd_tail_once = 0;
__device__ int g_bwd_cnt = 0;
__device__ int g_one_print = 0;
__device__ int g_nan_printed = 0;
__device__ int g_bad_px = -1;
__device__ int g_bad_py = -1;
__device__ int g_dbg_pix_lock = 0;
__device__ int g_dbg_pix_x = 226;
__device__ int g_dbg_pix_y = 221;

__device__ __forceinline__ void print_bad6(
    const char* tag, int idx,
    float a, float b, float c, float d, float e, float f)
{
    if ((!isfinite(a) || !isfinite(b) || !isfinite(c) ||
         !isfinite(d) || !isfinite(e) || !isfinite(f)) &&
        atomicCAS(&g_nan_printed, 0, 1) == 0)
    {
        printf("[%s] idx=%d a=%e b=%e c=%e d=%e e=%e f=%e\n",
               tag, idx, a, b, c, d, e, f);
    }
}

__device__ __forceinline__ void print_bad_if_needed(
    const char* tag, int gid, float a, float b, float c, float d)
{
    if ((!isfinite(a) || !isfinite(b) || !isfinite(c) || !isfinite(d)) &&
        atomicCAS(&g_nan_printed, 0, 1) == 0)
    {
        printf("[%s] gid=%d a=%e b=%e c=%e d=%e\n", tag, gid, a, b, c, d);
    }
}

__device__ __forceinline__ float sq(float x) { return x * x; }

// ===============================================================
// SH backward (unchanged from original)
// ===============================================================
__device__ void computeColorFromSH(
    int idx, int deg, int max_coeffs,
    const glm::vec3* means, glm::vec3 campos,
    const float* shs, const bool* clamped,
    const glm::vec3* dL_dcolor,
    glm::vec3* dL_dmeans,
    glm::vec3* dL_dshs)
{
    glm::vec3 pos = means[idx];
    glm::vec3 dir_orig = pos - campos;
    glm::vec3 dir = dir_orig / glm::length(dir_orig);

    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

    glm::vec3 dL_dRGB = dL_dcolor[idx];
    dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
    dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
    dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

    glm::vec3 dRGBdx(0, 0, 0);
    glm::vec3 dRGBdy(0, 0, 0);
    glm::vec3 dRGBdz(0, 0, 0);
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

    float dRGBdsh0 = SH_C0;
    dL_dsh[0] = dRGBdsh0 * dL_dRGB;

    if (deg > 0)
    {
        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 =  SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;
        dL_dsh[1] = dRGBdsh1 * dL_dRGB;
        dL_dsh[2] = dRGBdsh2 * dL_dRGB;
        dL_dsh[3] = dRGBdsh3 * dL_dRGB;

        dRGBdx = -SH_C1 * sh[3];
        dRGBdy = -SH_C1 * sh[1];
        dRGBdz =  SH_C1 * sh[2];

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);
            dL_dsh[4] = dRGBdsh4 * dL_dRGB;
            dL_dsh[5] = dRGBdsh5 * dL_dRGB;
            dL_dsh[6] = dRGBdsh6 * dL_dRGB;
            dL_dsh[7] = dRGBdsh7 * dL_dRGB;
            dL_dsh[8] = dRGBdsh8 * dL_dRGB;

            dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f *  x * sh[8];
            dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
            dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f *  z * sh[6] + SH_C2[3] * x * sh[7];

            if (deg > 2)
            {
                float dRGBdsh9  = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                dL_dsh[9]  = dRGBdsh9  * dL_dRGB;
                dL_dsh[10] = dRGBdsh10 * dL_dRGB;
                dL_dsh[11] = dRGBdsh11 * dL_dRGB;
                dL_dsh[12] = dRGBdsh12 * dL_dRGB;
                dL_dsh[13] = dRGBdsh13 * dL_dRGB;
                dL_dsh[14] = dRGBdsh14 * dL_dRGB;
                dL_dsh[15] = dRGBdsh15 * dL_dRGB;

                dRGBdx += (
                    SH_C3[0] * sh[9]  * 3.f * 2.f * xy +
                    SH_C3[1] * sh[10] * yz +
                    SH_C3[2] * sh[11] * -2.f * xy +
                    SH_C3[3] * sh[12] * -3.f * 2.f * xz +
                    SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
                    SH_C3[5] * sh[14] * 2.f * xz +
                    SH_C3[6] * sh[15] * 3.f * (xx - yy));

                dRGBdy += (
                    SH_C3[0] * sh[9]  * 3.f * (xx - yy) +
                    SH_C3[1] * sh[10] * xz +
                    SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
                    SH_C3[3] * sh[12] * -3.f * 2.f * yz +
                    SH_C3[4] * sh[13] * -2.f * xy +
                    SH_C3[5] * sh[14] * -2.f * yz +
                    SH_C3[6] * sh[15] * -3.f * 2.f * xy);

                dRGBdz += (
                    SH_C3[1] * sh[10] * xy +
                    SH_C3[2] * sh[11] * 4.f * 2.f * yz +
                    SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
                    SH_C3[4] * sh[13] * 4.f * 2.f * xz +
                    SH_C3[5] * sh[14] * (xx - yy));
            }
        }
    }

    glm::vec3 dL_ddir(
        glm::dot(dRGBdx, dL_dRGB),
        glm::dot(dRGBdy, dL_dRGB),
        glm::dot(dRGBdz, dL_dRGB));

    float3 dL_dmean = dnormvdv(
        float3{ dir_orig.x, dir_orig.y, dir_orig.z },
        float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

    dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// ===============================================================
// Cov2D inverse backward kernel (unchanged from original)
// ===============================================================
__global__ void computeCov2DCUDA(
    int P,
    const float3* means,
    const int* radii,
    const float* cov3Ds,
    const float h_x, float h_y,
    const float tan_fovx, float tan_fovy,
    const float* view_matrix,
    const float* opacities,
    const float* dL_dconics,
    float* dL_dopacity,
    const float* dL_dinvdepth,
    float3* dL_dmeans,
    float* dL_dcov,
    bool antialiasing)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    const float* cov3D = cov3Ds + 6 * idx;

    float3 mean = means[idx];
    float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
    float3 t = transformPoint4x3(mean, view_matrix);

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
    const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

    glm::mat3 J = glm::mat3(
        h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
        0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
        0, 0, 0);

    glm::mat3 W = glm::mat3(
        view_matrix[0], view_matrix[4], view_matrix[8],
        view_matrix[1], view_matrix[5], view_matrix[9],
        view_matrix[2], view_matrix[6], view_matrix[10]);

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 T = W * J;
    glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

    float c_xx = cov2D[0][0];
    float c_xy = cov2D[0][1];
    float c_yy = cov2D[1][1];

    constexpr float h_var = 0.3f;
    float d_inside_root = 0.f;
    if (antialiasing)
    {
        const float det_cov = c_xx * c_yy - c_xy * c_xy;
        c_xx += h_var;
        c_yy += h_var;
        const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
        const float h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov));
        const float dL_dopacity_v = dL_dopacity[idx];
        const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
        dL_dopacity[idx] = dL_dopacity_v * h_convolution_scaling;
        d_inside_root = (det_cov / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
    }
    else
    {
        c_xx += h_var;
        c_yy += h_var;
    }

    float dL_dc_xx = 0;
    float dL_dc_xy = 0;
    float dL_dc_yy = 0;
    if (antialiasing)
    {
        const float x = c_xx;
        const float y = c_yy;
        const float z = c_xy;
        const float w = h_var;
        const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
        const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
        const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
        const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
        dL_dc_xx = dL_dx;
        dL_dc_yy = dL_dy;
        dL_dc_xy = dL_dz;
    }

    float denom = c_xx * c_yy - c_xy * c_xy;
    float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

    print_bad6("BAD_cov2D_denom", idx, denom, denom2inv, c_xx, c_xy, c_yy, dL_dopacity[idx]);

    if (denom2inv != 0)
    {
        dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
        dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
        dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);

        print_bad6("BAD_cov2D_grad", idx, dL_dc_xx, dL_dc_xy, dL_dc_yy, dL_dconic.x, dL_dconic.y, dL_dconic.z);

        dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_dc_xx + T[0][0] * T[1][0] * dL_dc_xy + T[1][0] * T[1][0] * dL_dc_yy);
        dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_dc_xx + T[0][1] * T[1][1] * dL_dc_xy + T[1][1] * T[1][1] * dL_dc_yy);
        dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_dc_xx + T[0][2] * T[1][2] * dL_dc_xy + T[1][2] * T[1][2] * dL_dc_yy);

        dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_dc_xx + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][1] * dL_dc_yy;
        dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_dc_xx + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][2] * dL_dc_yy;
        dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_dc_xx + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_dc_xy + 2 * T[1][1] * T[1][2] * dL_dc_yy;
    }
    else
    {
        for (int i = 0; i < 6; i++) dL_dcov[6 * idx + i] = 0;
    }

    float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xx +
                    (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_xy;
    float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xx +
                    (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_xy;
    float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xx +
                    (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_xy;
    float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_yy +
                    (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xy;
    float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_yy +
                    (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xy;
    float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_yy +
                    (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xy;

    float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
    float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
    float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
    float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

    float tz = 1.f / t.z;
    float tz2 = tz * tz;
    float tz3 = tz2 * tz;

    float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
    float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
    float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11
                 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

    if (dL_dinvdepth) dL_dtz -= dL_dinvdepth[idx] / (t.z * t.z);

    float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

    print_bad6("BAD_cov2D_mean", idx, dL_dtx, dL_dty, dL_dtz, dL_dmean.x, dL_dmean.y, dL_dmean.z);

    dL_dmeans[idx] = dL_dmean;
}

// ===============================================================
// Cov3D backward (unchanged from original)
// ===============================================================
__device__ void computeCov3D(
    int idx,
    const glm::vec3 scale, float mod,
    const glm::vec4 rot,
    const float* dL_dcov3Ds,
    glm::vec3* dL_dscales,
    glm::vec4* dL_drots)
{
    glm::vec4 q = rot;
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 S = glm::mat3(1.0f);
    glm::vec3 s = mod * scale;
    S[0][0] = s.x;
    S[1][1] = s.y;
    S[2][2] = s.z;

    glm::mat3 M = S * R;

    const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

    glm::mat3 dL_dSigma = glm::mat3(
        dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
        0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
        0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
    );

    glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

    glm::mat3 Rt = glm::transpose(R);
    glm::mat3 dL_dMt = glm::transpose(dL_dM);

    glm::vec3* dL_dscale = dL_dscales + idx;
    dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
    dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
    dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

    dL_dMt[0] *= s.x;
    dL_dMt[1] *= s.y;
    dL_dMt[2] *= s.z;

    glm::vec4 dL_dq;
    dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
    dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
    dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
    dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

    float4* dL_drot = (float4*)(dL_drots + idx);
    *dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}

// ===============================================================
// preprocess backward (unchanged from original)
// ===============================================================
template<int C>
__global__ void preprocessCUDA(
    int P, int D, int M,
    const float3* means,
    const int* radii,
    const float* shs,
    const bool* clamped,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float* proj,
    const glm::vec3* campos,
    const float3* dL_dmean2D,
    glm::vec3* dL_dmeans,
    float* dL_dcolor,
    float* dL_dcov3D,
    float* dL_dsh,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot,
    float* dL_dopacity)
{
    auto idx = cg::this_grid().thread_rank();

    //if (idx == 0) printf("[PRE_BWD] D=%d M=%d shs_ptr=%d\n", D, M, shs != nullptr);

    if (idx >= P || !(radii[idx] > 0))
        return;

    float3 m = means[idx];

    float4 m_hom = transformPoint4x4(m, proj);
    float m_w = 1.0f / (m_hom.w + 0.0000001f);

    glm::vec3 dL_dmean;
    float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
    float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
    dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
    dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
    dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

    dL_dmeans[idx] += dL_dmean;

    if (shs)
        computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped,
                           (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

    if (scales)
        computeCov3D(idx, scales[idx], scale_modifier, rotations[idx],
                     dL_dcov3D, dL_dscale, dL_drot);
}

/*
// ===============================================================
// NEW: Warp-per-pixel backward render kernel (matches your forward)
// - blockDim.x must be 256 (8 warps)
// - each warp processes one pixel in the tile
// - lane0 performs the sequential recurrence for that pixel
// - all lanes help prefetch 32-gaussian chunks (reverse order)
// ===============================================================
template <uint32_t C>
__global__ void renderCUDA_warpPerPixel_bwd(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float* __restrict__ bg_color,
    const float2* __restrict__ points_xy_image,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ colors,
    const float* __restrict__ depths,
    const float* __restrict__ final_Ts,
    const uint32_t* __restrict__ n_contrib,
    const float* __restrict__ dL_dpixels,
    const float* __restrict__ dL_invdepths,
    float3* __restrict__ dL_dmean2D,
    float4* __restrict__ dL_dconic2D,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dcolors,
    float* __restrict__ dL_dinvdepths)
{
    const int WARPS_PER_BLOCK = 8;
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;

    if (blockDim.x != WARPS_PER_BLOCK * 32) return;

    const int tile_x = (int)blockIdx.x;
    const int tile_y = (int)blockIdx.y;

    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint32_t vertical_blocks   = (H + BLOCK_Y - 1) / BLOCK_Y;
    if ((uint32_t)tile_x >= horizontal_blocks || (uint32_t)tile_y >= vertical_blocks) return;

    const uint2 range = ranges[tile_y * horizontal_blocks + tile_x];
    const int toDo_total = (int)(range.y - range.x);

    // Per-warp shared chunk cache: each warp gets its own 32-entry slice.
    __shared__ int    sh_gid[WARPS_PER_BLOCK * 32];
    __shared__ float2 sh_xy[WARPS_PER_BLOCK * 32];
    __shared__ float4 sh_cono[WARPS_PER_BLOCK * 32];
    __shared__ float  sh_col[WARPS_PER_BLOCK * 32 * C];
    __shared__ float  sh_depth[WARPS_PER_BLOCK * 32];

    const int PASSES = (BLOCK_SIZE + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK; // 256/8 = 32

    for (int pass = 0; pass < PASSES; ++pass)
    {
        const int pixel_linear_in_tile = pass * WARPS_PER_BLOCK + warpId;
        if (pixel_linear_in_tile >= BLOCK_SIZE)
        {
            __syncthreads();
            continue;
        }

        const int local_px = pixel_linear_in_tile % BLOCK_X;
        const int local_py = pixel_linear_in_tile / BLOCK_X;

        const int pix_x = tile_x * BLOCK_X + local_px;
        const int pix_y = tile_y * BLOCK_Y + local_py;

        const bool inside = (pix_x < W && pix_y < H);
        const uint32_t pix_id = (uint32_t)(W * pix_y + pix_x);
        const float2 pixf = { (float)pix_x, (float)pix_y };

        float T_final = 0.0f;
        uint32_t last_contributor = 0;
        float T = 0.0f;

        float dL_dpixel_local[C];
        float dL_invdepth_local = 0.0f;

        if (lane == 0)
        {
            if (inside)
            {
                T_final = final_Ts[pix_id];
                last_contributor = n_contrib[pix_id];

                for (int ch = 0; ch < (int)C; ++ch)
                    dL_dpixel_local[ch] = dL_dpixels[ch * H * W + pix_id];

                if (dL_invdepths)
                    dL_invdepth_local = dL_invdepths[pix_id];
            }
            T = T_final;
        }

        // Broadcast lane0 state to the warp
        T_final = __shfl_sync(0xffffffffu, T_final, 0);
        last_contributor = (uint32_t)__shfl_sync(0xffffffffu, (int)last_contributor, 0);
        T = __shfl_sync(0xffffffffu, T, 0);

        if (!inside)
        {
            __syncthreads();
            continue;
        }

        float accum_rec[C];
        float last_color[C];
        float last_alpha = 0.0f;

        float accum_invdepth_rec = 0.0f;
        float last_invdepth = 0.0f;

        if (lane == 0)
        {
            for (int ch = 0; ch < (int)C; ++ch)
            {
                accum_rec[ch] = 0.0f;
                last_color[ch] = 0.0f;
            }
        }

        const float ddelx_dx = 0.5f * W;
        const float ddely_dy = 0.5f * H;

        // Same contributor semantics as original backward
        uint32_t contributor = (uint32_t)toDo_total;
        uint32_t used = 0;

        // Per-warp shared-memory slice base
        const int warp_base = warpId * 32;

        // Reverse traversal in 32-sized chunks
        for (int offset = 0; offset < toDo_total; offset += 32)
        {
            // Each warp prefetches ITS OWN reversed chunk into ITS OWN shared slice
            const int k = offset + lane;
            const int s = warp_base + lane;

            if (k < toDo_total)
            {
                const uint32_t idx_in_list = (uint32_t)(range.y - 1u - (uint32_t)k);
                const int gid = (int)point_list[idx_in_list];

                sh_gid[s]  = gid;
                sh_xy[s]   = points_xy_image[gid];
                sh_cono[s] = conic_opacity[gid];

                for (int ch = 0; ch < (int)C; ++ch)
                    sh_col[s * C + ch] = colors[gid * C + ch];

                sh_depth[s] = (dL_invdepths != nullptr) ? depths[gid] : 1.0f;
            }
            else
            {
                sh_gid[s]  = -1;
                sh_xy[s]   = make_float2(0.f, 0.f);
                sh_cono[s] = make_float4(0.f, 0.f, 0.f, 0.f);

                for (int ch = 0; ch < (int)C; ++ch)
                    sh_col[s * C + ch] = 0.0f;

                sh_depth[s] = 1.0f;
            }

            __syncwarp();
            __syncthreads();

            const int chunk_len = min(32, toDo_total - offset);

            // Exact original sequential replay on lane0 only
            if (lane == 0)
            {
                int dbg_seen = 0;

                if (lane == 0 && inside && pix_x == g_dbg_pix_x && pix_y == g_dbg_pix_y) {
                    //printf("[BWD_INIT] T_final=%e last=%u todo_total=%u\n", T_final, last_contributor, toDo_total);
                }

                for (int j = 0; j < chunk_len; ++j)
                {
                    contributor--;

                    // Same skip condition as original backward
                    if (contributor >= last_contributor)
                        continue;

                    const int sj = warp_base + j;
                    const int global_id = sh_gid[sj];
                    if (global_id < 0)
                        continue;

                    const float2 xy = sh_xy[sj];
                    const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
                    const float4 con_o = sh_cono[sj];

                    const float power =
                        -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y)
                        - con_o.y * d.x * d.y;

                    if (power > 0.0f)
                        continue;

                    const float G = expf(power);
                    const float alpha = fminf(0.99f, con_o.w * G);

                    print_bad_if_needed("BAD_ALPHA", global_id, alpha, G, power, con_o.w);

                    if (alpha < (1.0f / 255.0f))
                        continue;

                    used++;

                    if (pix_x == g_dbg_pix_x && pix_y == g_dbg_pix_y && dbg_seen < 8) {
                        float T_next_lin = T + alpha * T;
                        float T_next_exact = T / fmaxf(1.f - alpha, 1e-6f);
                        //printf("[BWD_HEAD] contrib=%u gid=%d alpha=%e T=%e T_lin=%e T_exact=%e last=%u\n", contributor, global_id, alpha, T, T_next_lin, T_next_exact, last_contributor);
                        dbg_seen++;
                    }


                    // Undo forward transmittance update exactly like original backward
                    //T = T / (1.f - alpha);
                    T = T + alpha * T;   // numerically stable inverse
                    //const float om = fmaxf(1.f - alpha, 1e-6f);
                    //T = T / om;

                    if (pix_x == g_dbg_pix_x && pix_y == g_dbg_pix_y) {
                        if (contributor >= last_contributor - 6 && contributor <= last_contributor) {
                            //printf("[BWD_TAIL] contrib=%u gid=%d alpha=%e T_after=%e last=%u\n", contributor, global_id, alpha, T, last_contributor);
                        }
                    }

                    if ((T > 1e3f || !isfinite(T)) && atomicCAS(&g_dbg_pix_lock, 0, 1) == 0)
                    {
                        g_dbg_pix_x = pix_x;
                        g_dbg_pix_y = pix_y;
                        printf("[T_EXPLODE] pix=(%d,%d) gid=%d alpha=%e T=%e T_final=%e contributor=%u last=%u\n",
                               pix_x, pix_y, global_id, alpha, T, T_final, contributor, last_contributor);
                    }

                    print_bad_if_needed("BAD_T", global_id, T, alpha, T_final, (float)last_contributor);

                    const float dchannel_dcolor = alpha * T;

                    float dL_dalpha = 0.0f;

                    for (int ch = 0; ch < (int)C; ++ch)
                    {
                        const float c = sh_col[sj * C + ch];

                        accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
                        last_color[ch] = c;

                        const float dL_dchannel = dL_dpixel_local[ch];
                        dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;

                        atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
                    }

                    if (dL_dinvdepths)
                    {
                        const float invd = 1.f / sh_depth[sj];
                        accum_invdepth_rec = last_alpha * last_invdepth + (1.f - last_alpha) * accum_invdepth_rec;
                        last_invdepth = invd;

                        dL_dalpha += (invd - accum_invdepth_rec) * dL_invdepth_local;
                        atomicAdd(&(dL_dinvdepths[global_id]), dchannel_dcolor * dL_invdepth_local);
                    }

                    dL_dalpha *= T;
                    last_alpha = alpha;

                    float bg_dot_dpixel = 0.0f;
                    for (int ch = 0; ch < (int)C; ++ch)
                        bg_dot_dpixel += bg_color[ch] * dL_dpixel_local[ch];

                    dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

                    print_bad_if_needed("BAD_dL_dalpha", global_id, dL_dalpha, T, alpha, bg_dot_dpixel);

                    const float dL_dG = con_o.w * dL_dalpha;
                    const float gdx = G * d.x;
                    const float gdy = G * d.y;
                    const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
                    const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

                    print_bad_if_needed("BAD_geom", global_id, dL_dG, dG_ddelx, dG_ddely, G);

                    atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
                    atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

                    float add_x = -0.5f * gdx * d.x * dL_dG;
                    float add_y = -0.5f * gdx * d.y * dL_dG;
                    float add_w = -0.5f * gdy * d.y * dL_dG;

                    if ((!isfinite(add_x) || !isfinite(add_y) || !isfinite(add_w) ||
                         fabsf(add_x) > 1e10f || fabsf(add_y) > 1e10f || fabsf(add_w) > 1e10f) &&
                        atomicCAS(&g_nan_printed, 0, 1) == 0)
                    {
                        printf("[BAD_render_conic] gid=%d alpha=%e G=%e dL_dalpha=%e dL_dG=%e dx=%e dy=%e addx=%e addy=%e addw=%e T=%e T_final=%e\n",
                               global_id, alpha, G, dL_dalpha, dL_dG, d.x, d.y,
                               add_x, add_y, add_w, T, T_final);
                    }

                    atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
                    atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
                    atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

                    atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
                }

                if (pix_x == g_dbg_pix_x && pix_y == g_dbg_pix_y) {
                    if(contributor >= last_contributor - 8 && contributor <= last_contributor) {
                        //printf("[REPLAY] last=%u contributor=%d\n", last_contributor, contributor);
                    }
                }

            }

            __syncwarp();
            __syncthreads();
        }
    }
}
*/

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_invdepths,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dinvdepths
)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];


	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float dL_invdepth;
	float accum_invdepth_rec = 0;
	if (inside)
	{
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		if(dL_invdepths)
		dL_invdepth = dL_invdepths[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_invdepth = 0;


	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];

			if(dL_invdepths)
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			// Propagate gradients from inverse depth to alphaas and
			// per Gaussian inverse depths
			if (dL_dinvdepths)
			{
			const float invd = 1.f / collected_depths[j];
			accum_invdepth_rec = last_alpha * last_invdepth + (1.f - last_alpha) * accum_invdepth_rec;
			last_invdepth = invd;
			dL_dalpha += (invd - accum_invdepth_rec) * dL_invdepth;
			atomicAdd(&(dL_dinvdepths[global_id]), dchannel_dcolor * dL_invdepth);
			}

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}



// ===============================================================
// Public API (BACKWARD namespace) - preprocess unchanged
// BACKWARD::render now launches warp-per-pixel kernel with block=256x1
// ===============================================================
void BACKWARD::preprocess(
    int P, int D, int M,
    const float3* means3D,
    const int* radii,
    const float* shs,
    const bool* clamped,
    const float* opacities,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const float* viewmatrix,
    const float* projmatrix,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    const glm::vec3* campos,
    const float3* dL_dmean2D,
    const float* dL_dconic,
    const float* dL_dinvdepth,
    float* dL_dopacity,
    glm::vec3* dL_dmean3D,
    float* dL_dcolor,
    float* dL_dcov3D,
    float* dL_dsh,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot,
    bool antialiasing)
{

    g_nan_printed = 0;

    computeCov2DCUDA<<< (P + 255) / 256, 256 >>>(
        P,
        means3D,
        radii,
        cov3Ds,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
        viewmatrix,
        opacities,
        dL_dconic,
        dL_dopacity,
        dL_dinvdepth,
        (float3*)dL_dmean3D,
        dL_dcov3D,
        antialiasing);

    preprocessCUDA<NUM_CHANNELS><<< (P + 255) / 256, 256 >>>(
        P, D, M,
        (float3*)means3D,
        radii,
        shs,
        clamped,
        (glm::vec3*)scales,
        (glm::vec4*)rotations,
        scale_modifier,
        projmatrix,
        campos,
        (float3*)dL_dmean2D,
        (glm::vec3*)dL_dmean3D,
        dL_dcolor,
        dL_dcov3D,
        dL_dsh,
        dL_dscale,
        dL_drot,
        dL_dopacity);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_invdepths,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dinvdepths)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_invdepths,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dinvdepths
		);
}

//void BACKWARD::render(
    //const dim3 grid, const dim3 /*block_ignored*/,
    /*const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float* bg_color,
    const float2* means2D,
    const float4* conic_opacity,
    const float* colors,
    const float* depths,
    const float* final_Ts,
    const uint32_t* n_contrib,
    const float* dL_dpixels,
    const float* dL_invdepths,
    float3* dL_dmean2D,
    float4* dL_dconic2D,
    float* dL_dopacity,
    float* dL_dcolors,
    float* dL_dinvdepths)
{
    (void)means2D; // not used in backward render (same as original kernel usage pattern)

    const int WARPS_PER_BLOCK = 8;
    const dim3 grid2(grid.x, grid.y, 1);
    const dim3 block2(32 * WARPS_PER_BLOCK, 1, 1); // 256 threads

    renderCUDA_warpPerPixel_bwd<NUM_CHANNELS><<< grid2, block2 >>>(
        ranges,
        point_list,
        W, H,
        bg_color,
        means2D,
        conic_opacity,
        colors,
        depths,
        final_Ts,
        n_contrib,
        dL_dpixels,
        dL_invdepths,
        dL_dmean2D,
        dL_dconic2D,
        dL_dopacity,
        dL_dcolors,
        dL_dinvdepths);

}*/

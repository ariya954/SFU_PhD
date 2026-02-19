/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
#include <cstdio>
 */
#include <cstdio>
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float3 normalize_f3(const float3& v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z) + 1e-8f;
    return make_float3(v.x/len, v.y/len, v.z/len);
}

__device__ __forceinline__ float dot_f3(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix) {
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;

	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// ------------------------------
// Warp-per-pixel forward render with warp scan (prefix product)
// ------------------------------

template <uint32_t CHANNELS>
__global__ void renderCUDA_warpPerPixel(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    const float* __restrict__ depths,
    float* __restrict__ invdepth)
{
    // ---- Tunables ----
    const int WARPS_PER_BLOCK = 8;
    const int lane  = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;

    // Guard: launch must match assumption
    if (blockDim.x != WARPS_PER_BLOCK * 32) return;

    // Decode tile
    const int tile_x = (int)blockIdx.x;
    const int BLOCKS_PER_TILE = (BLOCK_SIZE + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK; // 256/8=32
    const int tile_y = (int)(blockIdx.y / BLOCKS_PER_TILE);
    const int subblock = (int)(blockIdx.y - tile_y * BLOCKS_PER_TILE);

    // Warp -> pixel in tile
    const int pixel_linear_in_tile = subblock * WARPS_PER_BLOCK + warpId;
    if (pixel_linear_in_tile >= BLOCK_SIZE) return;

    const int local_px = pixel_linear_in_tile % BLOCK_X;
    const int local_py = pixel_linear_in_tile / BLOCK_X;

    const int pix_x = tile_x * BLOCK_X + local_px;
    const int pix_y = tile_y * BLOCK_Y + local_py;

    // Tile bounds (ranges indexing safety)
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint32_t vertical_blocks   = (H + BLOCK_Y - 1) / BLOCK_Y;
    if ((uint32_t)tile_x >= horizontal_blocks || (uint32_t)tile_y >= vertical_blocks) return;

    // Pixel bounds
    if (pix_x >= W || pix_y >= H) return;

    const uint32_t pix_id = (uint32_t)(W * pix_y + pix_x);
    const float2 pixf = { (float)pix_x, (float)pix_y };

    const uint2 range = ranges[tile_y * horizontal_blocks + tile_x];

    //if (pix_x == 528 && pix_y == 0 && lane == 0)
    //{
        //printf("[PIXDBG] pix=(%d,%d) pix_id=%u range=(%u,%u)\n",
               //pix_x, pix_y, pix_id, range.x, range.y);
    //}

    float T = 1.0f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    float C[CHANNELS] = {0};
    float expected_invdepth = 0.0f;

    const float T_EPS = 0.0001f;
    //int i = 0;
    // Process in full-warp chunks, but make invalid lanes neutral
    for (uint32_t base = range.x; base < range.y; base += 32)
    {

        //if (lane == 0) printf("lane0 reached checkpoint\n");

        unsigned full = 0xffffffffu;
        //i++; //printf("ALIVE=%08x base=%u\n", alive, base);
        //if (full == 0x00000001u) {
            //printf("[ALIVE] base=%u alive=%08x pix_id=%u warpTid=%u\n",
                   //(unsigned)base, full, pix_id, (unsigned)threadIdx.x);
        //}
        //if (lane == 0 && alive != 0xffffffffu) {
            //printf("[DROPOUT_TOP] pix_id=%u base=%u alive=%08x T=%e\n", pix_id, base, alive, T);
        //}
        //if ((threadIdx.x & 31) == 0 && alive != 0xffffffffu) {
            //printf("[FIRST_DROPOUT] pix_id=%u base=%u alive=%08x\n", pix_id, base, alive);
        //}

        // how many gaussians remain in this tile range for this chunk
        const uint32_t rem = range.y - base;
        const int chunk_len = (rem < 32u) ? (int)rem : 32;
        //alive = __activemask();
        const unsigned mask = (chunk_len == 32) ? full : ((1u << chunk_len) - 1u); //__ballot_sync(full, lane < chunk_len);

        // each lane's gaussian index
        const uint32_t k = base + (uint32_t)lane;

        // Default / neutral values for inactive lanes
        float alpha = 0.0f;
        float one_minus = 1.0f;

        float invd = 0.0f;

        float feat[CHANNELS];
        #pragma unroll
        for (int ch = 0; ch < (int)CHANNELS; ch++) feat[ch] = 0.0f;

        // Only lanes that correspond to a valid gaussian load real data
        if (lane < chunk_len)
        {
            const uint32_t gid = point_list[k];
            const float2 xy = points_xy_image[gid];
            const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            const float4 con_o = conic_opacity[gid];

            const float power =
                -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

            //if ((int)gid < 0 || gid >= 136029) {
                //if (lane == 0) {
                    //printf("[BADGID] pix_id=%u base=%u k=%u gid=%u range=(%u,%u)\n",
                           //pix_id, (unsigned)base, (unsigned)k, (unsigned)gid, range.x, range.y);
                //}
                //return; // or continue; for debug
            //}


            if (power <= 0.0f)
            {
                const float G = expf(power);
                alpha = fminf(0.99f, con_o.w * G);

                if (alpha >= (1.0f / 255.0f))
                {
                    #pragma unroll
                    for (int ch = 0; ch < (int)CHANNELS; ch++)
                        feat[ch] = features[gid * CHANNELS + ch];

                    //if (!isfinite(feat[2]) || fabsf(feat[2]) > 1e10f) {
                        //printf("[BADFEAT2] lane=%d pix_id=%u base=%u gid=%u feat2=%e feat0=%e feat1=%e\n",
                               //lane, pix_id, (unsigned)base, (unsigned)gid, feat[2], feat[0], feat[1]);
                    //}

                    //bool bad = false;
                    //#pragma unroll
                    //for (int ch = 0; ch < (int)CHANNELS; ch++) {
                        //bad |= !isfinite(feat[ch]);
                    //}
                    //bad |= !isfinite(alpha);

                    //if (bad) {
                        //printf("[BADFEAT] pix_id=%u base=%u lane=%d k=%u gid=%u alpha=%e feat=(%e,%e,%e)\n",
                               //pix_id, base, lane, k, gid, alpha, feat[0], feat[1], feat[2]);
                    //}


                    if (invdepth) invd = 1.0f / depths[gid];
                }
                else
                {
                    alpha = 0.0f;
                }
            }

            // After alpha is computed (and after alpha>=1/255 test)
            one_minus = 1.0f - alpha;
        }
        else
        {
            // inactive lanes: keep neutral
            alpha = 0.0f;
            one_minus = 1.0f;
        }

        // contributor counting: original style is "gaussians visited"
        const uint32_t contributor_base = contributor;
        contributor = contributor_base + (uint32_t)chunk_len;

        // Important: from this point on, use mask for ALL warp collectives
        __syncwarp(mask);

        // Inclusive product over 32 lanes (inactive lanes multiply by 1, so OK)
        float prefix_prod = warpInclusiveProd(one_minus, mask);

        //const float prefix_last = __shfl_sync(mask, prefix_prod, chunk_len - 1);

        //printf("prefix0=%e prefix_last=%e chunk_len=%d mask=%08x\n",
               //prefix_prod, prefix_last, chunk_len, mask);

        // Early termination detection (only meaningful where alpha>0)
        const float T_after_lane = T * prefix_prod;
        unsigned stop_mask = __ballot_sync(full, (alpha > 0.0f) && (T_after_lane < T_EPS));

        if (stop_mask)
        {
            const int stop_lane = __ffs(stop_mask) - 1;

            if (lane > stop_lane)
            {
                alpha = 0.0f;
                one_minus = 1.0f;
                #pragma unroll
                for (int ch = 0; ch < (int)CHANNELS; ch++) feat[ch] = 0.0f;
                invd = 0.0f;
            }

            // Recompute prefix after disabling lanes
            __syncwarp(mask);
            prefix_prod = warpInclusiveProd(one_minus, mask);
        }

        // Recompute prefix after disabling lanes
        //prefix_prod = warpInclusiveProd(one_minus, full);

        const bool lane_active = (lane < chunk_len);

        // Everyone in mask executes this (mask should match lane_active)
        //float excl_prod = (lane == 0) ? 1.f : __shfl_sync(full, prefix_prod, lane - 1);
        float excl_prod = __shfl_sync(full, prefix_prod, (lane > 0 ? lane - 1 : 0));
        if (lane == 0) excl_prod = 1.0f;      // correct exclusive prefix for lane0
        if (!lane_active) excl_prod = 1.0f;   // neutral for inactive lanes

        //float excl_prod = 1.0f;
        //if (lane < chunk_len) {
            //excl_prod = (lane == 0) ? 1.0f : __shfl_sync(mask, prefix_prod, lane - 1);
        //}
        //const float Tk = T * excl_prod;


        //const bool lane_active = (lane < chunk_len);
        //float excl_prod = 1.0f;

        //if (lane_active) {
            //if (lane == 0) excl_prod = 1.0f;
            //else          excl_prod = __shfl_sync(mask, prefix_prod, lane - 1);
        //}
        // else: keep excl_prod = 1.0f (neutral)


        //float excl_prod = (lane == 0) ? 1.0f : __shfl_sync(mask, prefix_prod, lane - 1);

        // else: keep excl_prod = 1.0f (neutral)

        //if (lane == 0) excl_prod = 1.0f;                        // lane0 fix

        //if (pix_id == 20612 && base == 26786 && (lane == 0 || lane == 1)) {
            //printf("[TKDBG] lane=%d T=%e prefix_prod=%e excl_prod=%e mask=%08x\n",
                   //lane, T, prefix_prod, excl_prod, mask);
        //}

        const float Tk = T * excl_prod;


        //if (pix_id == 20612 && base == 26786 && lane == 1) {
            //printf("[TKDBG2] Tk=%e isfinite(T)=%d isfinite(excl)=%d\n",
                   //Tk, isfinite(T), isfinite(excl_prod));
        //}

        //float local2 = feat[2] * alpha * Tk;

        //if (!isfinite(local2)) {
            //printf("[BADLOCAL2] lane=%d pix_id=%u base=%u alpha=%e Tk=%e feat2=%e\n",
                   //lane, pix_id, (unsigned)base, alpha, Tk, feat[2]);
        //}


        // Per-lane contributions
        float contribC[CHANNELS];
        #pragma unroll
        for (int ch = 0; ch < (int)CHANNELS; ch++)
            contribC[ch] = feat[ch] * alpha * Tk;

        float contribInvd = (invdepth) ? (invd * alpha * Tk) : 0.0f;

        //float local0 = contribC[0];

        //if (pix_x == 528 && pix_y == 0 && base == 11580 && lane == 0) {
            //for (int ch = 0; ch < (int)CHANNELS; ch++)
                //printf("[PRE] CHANNEL[%d]=%f\n", ch, contribC[ch]);
        //}

        // (1) local, before reduction
        //if (pix_x == 528 && pix_y == 0 && base == 11580 && lane == 0) {
            //printf("[PRE] alpha=%e Tk=%e feat0=%e local0=%e\n", alpha, Tk, feat[0], local0);
        //}

        //float red0 = warpReduceSum(local0, 0xffffffffu);

        // (2) after reduction
        //if (pix_x == 528 && pix_y == 0 && base == 11580 && lane == 0) {
            //printf("[POST] red0=%e\n", red0);
        //}

        // --- DEBUG: find which lane produces NaN/Inf before reduction ---
        //const unsigned FULL = mask;

        //float dbg_contrib0 = contribC[0];      // per-lane, before reduction
        //float dbg_feat0    = feat[0];

        //int bad = 0;
        //bad |= !isfinite(alpha);
        //bad |= !isfinite(Tk);
        //bad |= !isfinite(dbg_feat0);
        //bad |= !isfinite(dbg_contrib0);

        //unsigned badmask = __ballot_sync(mask, bad);

        //if (badmask && lane == 0 && pix_x == 528 && pix_y == 0) {
            //int first = __ffs(badmask) - 1;
            // pull the "bad lane" values into lane 0
           // float balpha   = __shfl_sync(mask, alpha, first);
           // float bTk      = __shfl_sync(mask, Tk, first);
           // float bfeat0   = __shfl_sync(mask, dbg_feat0, first);
           // float bcontrib = __shfl_sync(mask, dbg_contrib0, first);

            //printf("[BADLANE] base=%u first_lane=%d alpha=%e Tk=%e feat0=%e contrib0=%e chunk_len=%d\n",
                   //(unsigned)base, first, balpha, bTk, bfeat0, bcontrib, chunk_len);
        //}

        // only for ONE pixel and a few iterations or it will spam + slow to death
        //const bool dbg_pix = (pix_x == 528 && pix_y == 0);

        //float seqT = T;

        // Compare only at chunk start (choose one base or first few bases)
        //if (dbg_pix) {
            // sequential product of (1 - alpha) for the chunk
            //for (int l = 0; l < chunk_len; ++l) {
                //float a_l = __shfl_sync(full, alpha, l);       // ALL lanes execute
                //if (lane == 0) {
                    //seqT *= (1.0f - a_l);
                //}
            //}

            //float chunk_prod = __shfl_sync(full, prefix_prod, chunk_len - 1);  // ALL lanes execute
            //float warpNextT  = T * chunk_prod;

            //if (lane == 0) {
                //printf("[SEQvsWARP] base=%u chunk_len=%d seqNextT=%e warpNextT=%e chunk_prod=%e\n",
                       //(unsigned)base, chunk_len, seqT, warpNextT, chunk_prod);
            //}
        //}

        //if (lane == 0 && pix_x == 528 && pix_y == 0 && base == range.x) {
            //float seqT = T;               // T at chunk start
            //float seqC0 = 0.f, seqC1 = 0.f, seqC2 = 0.f;

            // gather alpha/feat from each lane (only first chunk_len lanes)
            //for (int l = 0; l < chunk_len; l++) {
                //float a  = __shfl_sync(mask, alpha, l);
                //float f0 = __shfl_sync(mask, feat[0], l);
                //float f1 = __shfl_sync(mask, feat[1], l);
                //float f2 = __shfl_sync(mask, feat[2], l);

                // sequential compositor
                //float Tk_seq = seqT;
                //seqC0 += f0 * a * Tk_seq;
                //seqC1 += f1 * a * Tk_seq;
                //seqC2 += f2 * a * Tk_seq;
                //seqT *= (1.0f - a);

                //if (seqT < T_EPS) break;
            //}

            // warp's chunk_prod and reduced contrib (what you're using)
            //float warp_chunk_prod = __shfl_sync(mask, prefix_prod, chunk_len - 1);
            //float warp_nextT = T * warp_chunk_prod;

            //printf("[SEQvsWARP] base=%u chunk_len=%d seqNextT=%e warpNextT=%e\n",
                   //(unsigned)base, chunk_len, seqT, warp_nextT);
        //}


        // Warp reduce sums (lane0 will use the sum)
        #pragma unroll
        for (int ch = 0; ch < (int)CHANNELS; ch++)
            contribC[ch] = warpReduceSum(contribC[ch], full);

        // Only debug the bad pixel + the chunk where it breaks
        //if (pix_x == 528 && pix_y == 0 && base == 11580)
        //{
            // check the "sources" that could poison the reduction
            //bool bad = false;
            //if (!isfinite(alpha) || !isfinite(Tk)) bad = true;
            //#pragma unroll
            //for (int ch = 0; ch < (int)CHANNELS; ch++)
                //if (!isfinite(feat[ch]) || !isfinite(contribC[ch])) bad = true;

            //unsigned bad_mask = __ballot_sync(mask, bad);

            //if (bad_mask)
            //{
                //int first = __ffs(bad_mask) - 1;
                //if (lane == first)
                //{
                    //printf("[NANLANE] lane=%d base=%u k=%u alpha=%e Tk=%e feat0=%e contrib0=%e\n",
                           //lane, base, (unsigned)k, alpha, Tk,
                           //feat[0], contribC[0]);
                //}
            //}
        //}

        contribInvd = warpReduceSum(contribInvd, full);

        // Last contributor (1-based index) among lanes that actually contributed
        unsigned contrib_mask = __ballot_sync(mask, (alpha > 0.0f));
        uint32_t chunk_last = 0;
        if (contrib_mask)
        {
            const int last_lane = 31 - __clz(contrib_mask);
            // But don't allow last_lane beyond chunk_len-1
            if (last_lane < chunk_len)
                chunk_last = contributor_base + (uint32_t)last_lane + 1u;
        }

        //if (lane == 0) printf("lane0 reached checkpoint\n");

        const float chunk_prod = __shfl_sync(full, prefix_prod, chunk_len - 1);

        // Lane0 commits
        if (lane == 0)
        {
            #pragma unroll
            for (int ch = 0; ch < (int)CHANNELS; ch++)
                C[ch] += contribC[ch];

            if (invdepth) expected_invdepth += contribInvd;

            if (chunk_last != 0) last_contributor = chunk_last;

            // Product of all lanes: lane31 is fine because inactive lanes are neutral (1.0)
            //const float chunk_prod = __shfl_sync(mask, prefix_prod, chunk_len - 1);

            //if (pix_x == 528 && pix_y == 0 && lane == 0 && (base == range.x || base == range.x + 32))
            //{
                //printf("[PRODDBG] base=%u T=%e chunk_prod=%e\n", base, T, chunk_prod);
            //}
            //printf("PREFIX_PROD =%f\n", prefix_prod);
            T *= chunk_prod;
        }

        //if (pix_x == 528 && pix_y == 0 && lane == 0)
        //{
            //printf("[OUTDBG] T=%e C0=%e C1=%e C2=%e last=%u\n",
                   //T, C[0], C[1], C[2], last_contributor);
        //}

        //alive = __activemask();
        //if (alive != 0xffffffffu && lane == 0) {
            //printf("[ACTMASK] pix_id=%u base=%u alive=%08x\n", pix_id, base, alive);
        //}
        //if (lane == 0) printf("[BEFORE_BREAK] pix_id=%u base=%u T(l0)=%e\n", pix_id, base, T);

        // Broadcast updated T/last_contributor
        T = __shfl_sync(full, T, 0);
        last_contributor = __shfl_sync(full, last_contributor, 0);

        //if (lane == 0 && pix_id == 528) {
            //if (!isfinite(T) || T < 0.0f || T > 1.0f) printf("[TBAD] T=%e base=%u\n", T, base);
        //}
        //if (lane == 0) printf("[AFTER_BCAST] pix_id=%u base=%u T(bcast)=%e excl_prod=%e lane=%d\n", pix_id, base, T, excl_prod, lane);

        //unsigned am = __activemask();
        //float Tb = __shfl_sync(am, T, 0);
        //printf("[XXXX_BCAST] lane=%d am=%08x T=%e Tb=%e\n", lane, am, T, Tb);

        //if (lane == 0) printf("lane0 reached checkpoint\n");

        //float T0 = __shfl_sync(mask, T, 0);
        //if (T != T0) printf("[TBCAST] pix_id=%u base=%u lane=%d T=%e T0=%e\n", pix_id, base, lane, T, T0);

        //if (mask & (1u << lane)) {
            //float T0 = __shfl_sync(mask, T, 0);
            //if (T != T0) {
                //printf("[TBCAST] pix_id=%u base=%u lane=%d T=%e T0=%e mask=%08x chunk_len=%d\n",
                       //pix_id, base, lane, T, T0, mask, chunk_len);
            //}
        //}

        //T0 = __shfl_sync(FULL, T, 0);
        //if ((threadIdx.x & 31) == 0 && T != T0) {
            //printf("[TDIVERGE] pix_id=%u base=%u T(l0)=%e T(lane0-shfl)=%e\n", pix_id, base, T, T0);
        //}

        // if (T < T_EPS) break;

        // uniform break
        int done = (lane == 0 && T < T_EPS);
        done = __shfl_sync(full, done, 0);
        if (done) break;

    }

    if (lane == 0)
    {
        final_T[pix_id] = T;
        n_contrib[pix_id] = last_contributor;

        #pragma unroll
        for (int ch = 0; ch < (int)CHANNELS; ch++)
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

        if (invdepth)
            invdepth[pix_id] = expected_invdepth;
    }
}

namespace FORWARD
{
	void render(
		const dim3 grid, dim3 /*block_ignored*/,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* means2D,
		const float* colors,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* depths,
		float* depth)
	{
		// We reinterpret the incoming "grid" as tile_grid (same as before),
		// and expand grid.y by BLOCKS_PER_TILE so one tile uses multiple blocks.
		const int WARPS_PER_BLOCK = 8;
		const int BLOCKS_PER_TILE = (BLOCK_SIZE + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK; // 256/8=32

		const dim3 grid2(grid.x, grid.y * BLOCKS_PER_TILE, 1);
		const dim3 block2(32 * WARPS_PER_BLOCK, 1, 1);

		renderCUDA_warpPerPixel<NUM_CHANNELS> <<< grid2, block2 >>>(
			ranges,
			point_list,
			W, H,
			means2D,
			colors,
			conic_opacity,
			final_T,
			n_contrib,
			bg_color,
			out_color,
			depths,
			depth
		);
	}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}

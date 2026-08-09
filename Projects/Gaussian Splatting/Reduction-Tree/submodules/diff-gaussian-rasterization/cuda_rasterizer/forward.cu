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

    const int WARPS_PER_BLOCK = 8;                  // 256 / 32

    const int tid =
        threadIdx.x +
        blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    const int lane   = tid & 31;
    const int warpId = tid >> 5;

    const int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    if (threads_per_block != WARPS_PER_BLOCK * 32) return;

    // Tile coordinates = one block per tile
    const int tile_x = (int)blockIdx.x;
    const int tile_y = (int)blockIdx.y;

    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint32_t vertical_blocks   = (H + BLOCK_Y - 1) / BLOCK_Y;
    if ((uint32_t)tile_x >= horizontal_blocks || (uint32_t)tile_y >= vertical_blocks) return;

    const uint2 range = ranges[tile_y * horizontal_blocks + tile_x];

    __shared__ uint32_t sh_gid[32];
    __shared__ float2   sh_xy[32];
    __shared__ float4   sh_cono[32];
    __shared__ float    sh_feat[32 * CHANNELS];
    __shared__ float    sh_invd[32];

    const int PASSES = (BLOCK_SIZE + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK; // 256/8 = 32

    for (int pass = 0; pass < PASSES; ++pass)
    {
        const int pixel_linear_in_tile = pass * WARPS_PER_BLOCK + warpId;
        if (pixel_linear_in_tile >= BLOCK_SIZE)
            continue;

        const int local_px = pixel_linear_in_tile % BLOCK_X;
        const int local_py = pixel_linear_in_tile / BLOCK_X;

        const int pix_x = tile_x * BLOCK_X + local_px;
        const int pix_y = tile_y * BLOCK_Y + local_py;

        const bool inside = (pix_x < W && pix_y < H);
        const uint32_t pix_id = (uint32_t)(W * pix_y + pix_x);
        const float2 pixf = { (float)pix_x, (float)pix_y };

        bool warp_done = !inside;

        // Per-pixel accumulators
        float T = 1.0f;
        uint32_t contributor = 0;
        uint32_t last_contributor = 0;

        float C[CHANNELS];
        #pragma unroll
        for (int ch = 0; ch < (int)CHANNELS; ++ch) C[ch] = 0.0f;

        float expected_invdepth = 0.0f;

        // Process this pixel's Gaussian list in chunks of 32
        for (uint32_t base = range.x; base < range.y; base += 32)
        {
            const uint32_t rem = range.y - base;
            const int chunk_len = (rem < 32u) ? (int)rem : 32;

            if (threadIdx.x < 32)
            {
                const int t = threadIdx.x;
                if (t < chunk_len)
                {
                    const uint32_t gid = point_list[base + (uint32_t)t];
                    sh_gid[t]  = gid;
                    sh_xy[t]   = points_xy_image[gid];
                    sh_cono[t] = conic_opacity[gid];

                    #pragma unroll
                    for (int ch = 0; ch < (int)CHANNELS; ++ch)
                        sh_feat[t * CHANNELS + ch] = features[gid * CHANNELS + ch];

                    sh_invd[t] = (invdepth != nullptr) ? (1.0f / depths[gid]) : 0.0f;

                }
                else
                {
                    sh_gid[t]  = 0;
                    sh_xy[t]   = make_float2(0.f, 0.f);
                    sh_cono[t] = make_float4(0.f, 0.f, 0.f, 0.f);

                    #pragma unroll
                    for (int ch = 0; ch < (int)CHANNELS; ++ch)
                        sh_feat[t * CHANNELS + ch] = 0.0f;

                    sh_invd[t] = 0.0f;

                }
            }
            __syncthreads();

            float alpha_lane = 0.0f;
            uint32_t gid_lane = 0;
            float2 xy = make_float2(0.f, 0.f);
            float4 con_o = make_float4(0.f, 0.f, 0.f, 0.f);

            if (!warp_done && lane < chunk_len)
            {
                gid_lane = sh_gid[lane];
                xy = sh_xy[lane];
                con_o = sh_cono[lane];

                const float2 d = { xy.x - pixf.x, xy.y - pixf.y };

                const float power =
                    -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y)
                    -       (con_o.y * d.x * d.y);

                if (power <= 0.0f)
                {
                    alpha_lane = fminf(0.99f, con_o.w * expf(power));
                    if (alpha_lane < (1.0f / 255.0f))
                        alpha_lane = 0.0f;
                }
            }

            const uint32_t contributor_base = contributor;
            const float T_before_chunk = T;

            const bool lane_active =
                (lane < chunk_len) &&
                (alpha_lane > 0.0f);

            const unsigned active_mask = __ballot_sync(0xffffffffu, lane_active);

            // Parallel stop-lane detection using full prefix product over all contributing lanes
            const float one_minus_all = lane_active ? (1.0f - alpha_lane) : 1.0f;

            // inclusive product over all contributing lanes in this chunk
            float prefix_prod_all = warpInclusiveProd(one_minus_all, 0xffffffffu);

            // transmission after applying this lane
            const float T_after_lane =
                lane_active ? (T_before_chunk * prefix_prod_all) : 1.0f;

            // first contributing lane that would make T drop below threshold
            unsigned stop_mask = __ballot_sync(
                0xffffffffu,
                lane_active && (T_after_lane < 0.0001f)
            );

            // Parallel bookkeeping from stop_lane and active lanes
            const int stop_lane_book = stop_mask ? (__ffs(stop_mask) - 1) : chunk_len;

            if (lane == 0 && !warp_done)
            {
                contributor = contributor_base + (uint32_t)((stop_lane_book < chunk_len) ? (stop_lane_book + 1) : chunk_len);

                unsigned before_stop_mask;
                if (stop_lane_book >= 32)
                    before_stop_mask = 0xffffffffu;
                else if (stop_lane_book == 0)
                    before_stop_mask = 0u;
                else
                    before_stop_mask = (1u << stop_lane_book) - 1u;

                unsigned used_mask = active_mask & before_stop_mask;

                if (used_mask)
                {
                    const int last_lane = 31 - __clz(used_mask);
                    last_contributor = contributor_base + (uint32_t)last_lane + 1u;
                }
            }

            const int stop_lane = stop_lane_book;
            // lane contributes if it is before stop and alpha is nonzero
            const bool lane_used =
                (lane < chunk_len) &&
                (alpha_lane > 0.0f) &&
                (lane < stop_lane);

            const float prefix_prod = prefix_prod_all;

            // exclusive prefix
            float excl_prod = __shfl_up_sync(0xffffffffu, prefix_prod, 1);
            if (lane == 0) excl_prod = 1.0f;

            // Tk = T_before_chunk * product of prior (1-alpha)
            const float Tk = lane_used ? (T_before_chunk * excl_prod) : 0.0f;
            const float w = alpha_lane * Tk;

            const int chunk_prod_lane = (stop_lane_book < chunk_len) ? stop_lane_book : 31;
            const float chunk_prod = __shfl_sync(0xffffffffu, prefix_prod_all, chunk_prod_lane);

            float contribC[CHANNELS];
            #pragma unroll
            for (int ch = 0; ch < (int)CHANNELS; ++ch)
                contribC[ch] = 0.0f;

            float contribInvd = 0.0f;

            if (lane_used)
            {
                #pragma unroll
                for (int ch = 0; ch < (int)CHANNELS; ++ch)
                    contribC[ch] = sh_feat[lane * CHANNELS + ch] * w;

                if (invdepth != nullptr)
                    contribInvd = sh_invd[lane] * w;
            }

            #pragma unroll
            for (int ch = 0; ch < (int)CHANNELS; ++ch)
                contribC[ch] = warpReduceSum(contribC[ch], 0xffffffffu);

            contribInvd = warpReduceSum(contribInvd, 0xffffffffu);

            if (lane == 0)
            {
                #pragma unroll
                for (int ch = 0; ch < (int)CHANNELS; ++ch)
                    C[ch] += contribC[ch];

                if (invdepth != nullptr)
                    expected_invdepth += contribInvd;

                T = T_before_chunk * chunk_prod;
            }

            T = __shfl_sync(0xffffffffu, T, 0);
            last_contributor = __shfl_sync(0xffffffffu, (int)last_contributor, 0);
            contributor = __shfl_sync(0xffffffffu, (int)contributor, 0);

            int done_now = __shfl_sync(0xffffffffu, (lane == 0 && !warp_done && T < 0.0001f) ? 1 : 0, 0);
            if (done_now) warp_done = true;

        }

        if (lane == 0 && inside)
        {
            final_T[pix_id] = T;
            n_contrib[pix_id] = last_contributor;

            #pragma unroll
            for (int ch = 0; ch < (int)CHANNELS; ++ch)
                out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

            if (invdepth != nullptr)
                invdepth[pix_id] = expected_invdepth;

        }

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
        const int WARPS_PER_BLOCK = 8;
        const dim3 grid2(grid.x, grid.y, 1);                 // <-- DO NOT expand grid.y
        const dim3 block2(256, 1, 1);       // 256 threads

        renderCUDA_warpPerPixel<NUM_CHANNELS><<< grid2, block2 >>>(
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

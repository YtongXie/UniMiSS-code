
render_kernel=r'''
texture<float, cudaTextureType3D, cudaReadModeElementType> t_volume;
texture<float, cudaTextureType2D, cudaReadModeElementType> t_proj_param_Nx12;

__device__ __constant__ float  d_step_size_mm;
__device__ __constant__ float3 d_image_size;
__device__ __constant__ float3 d_volume_spacing;
__device__ __constant__ float3 d_volume_corner_mm;

// negate
__device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// min
static __device__ float3 fminf(float3 a, float3 b)
{
	return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

// max
static __device__ float3 fmaxf(float3 a, float3 b)
{
	return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

// addition
static __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
__device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
static __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
__device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
__device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

static __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ void operator*=(float3 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
static __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
static __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
__device__ float3 operator/(float s, float3 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
__device__ void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// dot product
__device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// normalize
__device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// floor
static __device__ float3 floor(float3 v)
{
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};

__device__ Ray computeNormalizedRay(const float x, const float y, const int ch)
{
    // compute a unit vector connecting the source and the normalized pixel (x, y) on the imaging plane using pre-computed corner points.
    Ray ray;
    ray.d = normalize( make_float3( tex2D(t_proj_param_Nx12, 0, ch)+tex2D(t_proj_param_Nx12, 3, ch)*x+tex2D(t_proj_param_Nx12, 6, ch)*y,
                                    tex2D(t_proj_param_Nx12, 1, ch)+tex2D(t_proj_param_Nx12, 4, ch)*x+tex2D(t_proj_param_Nx12, 7, ch)*y,
                                    tex2D(t_proj_param_Nx12, 2, ch)+tex2D(t_proj_param_Nx12, 5, ch)*x+tex2D(t_proj_param_Nx12, 8, ch)*y ) );
    ray.o = make_float3( tex2D(t_proj_param_Nx12, 9, ch), tex2D(t_proj_param_Nx12, 10, ch), tex2D(t_proj_param_Nx12, 11, ch) );
    return ray;
}

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
// This code is based on volumeRender demo in CUDA SDK
__device__ bool intersectBoxRay(float3 box, const Ray ray, float &tnear, float &tfar)
{
    // compute intersection of ray with all six planes
    float3 tbot = (-box - ray.o) / ray.d;
    float3 ttop = ( box - ray.o) / ray.d;

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    tfar  = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
    return tfar > tnear;
}


__global__ void render_with_linear_interp(float *d_image_N)
{

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    int index = x*int(d_image_size.z)*int(d_image_size.y)+y*int(d_image_size.z)+z;
    //if (index >= (d_image_size.x * d_image_size.y * d_image_size.z)) return;
    if (index >= (d_image_size.x * d_image_size.y * d_image_size.z) ||
        (x >= d_image_size.x || y >= d_image_size.y || z >= d_image_size.z)
        )
        return;

    Ray ray = computeNormalizedRay(
        ((float)x+0.5f)/d_image_size.x,
        ((float)y+0.5f)/d_image_size.y,
        z
        );

    float tnear = 0.0f, tfar = 0.0f, RPL = 0.0f;
    if(!intersectBoxRay(d_volume_corner_mm, ray, tnear, tfar)) return;

    // compute Radiological Path Length (RPL) by trilinear interpolation (texture fetching)
    float3 cur = (ray.o + tnear * ray.d + d_volume_corner_mm) / d_volume_spacing;  // object coordinate (mm) -> texture (voxel) coordinate
    float3 delta_dir = d_step_size_mm * ray.d / d_volume_spacing;                  // object coordinate (mm) -> texture (voxel) coordinate

    for(float travelled_length = 0; travelled_length < (tfar-tnear); travelled_length += d_step_size_mm, cur += delta_dir){
        // pick the density value at the current point and accumulate it (Note: currently consider only single input volume case)
        // access to register memory and texture memory (filterMode of texture should be 'linear')
        RPL += tex3D(t_volume, cur.z, cur.y, cur.x) * d_step_size_mm;
    }

    d_image_N[index] += RPL;
}

__device__ float bspline(float t)
{
	t = fabs(t);
	const float a = 2.0f - t;

	if (t < 1.0f) return 2.0f/3.0f - 0.5f*t*t*a;
	else if (t < 2.0f) return a*a*a / 6.0f;
	else return 0.0f;
}

__device__ float cubicTex3D(texture<float, cudaTextureType3D, cudaReadModeElementType> tex, float i, float j, float k)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = make_float3(i, j, k) - 0.5f;
	float3 index = floor(coord_grid);
	const float3 fraction = coord_grid - index;
	index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

	float result = 0.0f;
	for (float z=-1; z < 2.5f; z++)  //range [-1, 2]
	{
		float bsplineZ = bspline(z-fraction.z);
		float w = index.z + z;
		for (float y=-1; y < 2.5f; y++)
		{
			float bsplineYZ = bspline(y-fraction.y) * bsplineZ;
			float v = index.y + y;
			for (float x=-1; x < 2.5f; x++)
			{
				float bsplineXYZ = bspline(x-fraction.x) * bsplineYZ;
				float u = index.x + x;
				result += bsplineXYZ * tex3D(tex, u, v, w);
			}
		}
	}
	return result;
}

__global__ void render_with_cubic_interp(float *d_image_N)
{

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    int index = x*int(d_image_size.z)*int(d_image_size.y)+y*int(d_image_size.z)+z;
    //if (index >= (d_image_size.x * d_image_size.y * d_image_size.z)) return;
    if (index >= (d_image_size.x * d_image_size.y * d_image_size.z) ||
        (x >= d_image_size.x || y >= d_image_size.y || z >= d_image_size.z)
        )
        return;

    Ray ray = computeNormalizedRay(
        ((float)x+0.5f)/d_image_size.x,
        ((float)y+0.5f)/d_image_size.y,
        z
        );

    float tnear = 0.0f, tfar = 0.0f, RPL = 0.0f;
    if(!intersectBoxRay(d_volume_corner_mm, ray, tnear, tfar)) return;

    // compute Radiological Path Length (RPL) by trilinear interpolation (texture fetching)
    float3 cur = (ray.o + tnear * ray.d + d_volume_corner_mm) / d_volume_spacing;  // object coordinate (mm) -> texture (voxel) coordinate
    float3 delta_dir = d_step_size_mm * ray.d / d_volume_spacing;                  // object coordinate (mm) -> texture (voxel) coordinate

    for(float travelled_length = 0; travelled_length < (tfar-tnear); travelled_length += d_step_size_mm, cur += delta_dir){
        // pick the density value at the current point and accumulate it (Note: currently consider only single input volume case)
        // access to register memory and texture memory (filterMode of texture should be 'linear')
        RPL += cubicTex3D(t_volume, cur.z, cur.y, cur.x) * d_step_size_mm;
    }

    d_image_N[index] += RPL;
}


// Debug print
__global__ void print_device_params()
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x != 0 || y != 0 || z != 0)
    {
        return;
    }

    printf("step_size_mm: %f\n", d_step_size_mm);
    printf("image_size: %f, %f, %f\n", d_image_size.x, d_image_size.y, d_image_size.z);
    printf("volume_spacing: %f, %f, %f\n", d_volume_spacing.x, d_volume_spacing.y, d_volume_spacing.z);
    printf("volume_corner_mm: %f, %f, %f\n", d_volume_corner_mm.x, d_volume_corner_mm.y, d_volume_corner_mm.z);

    for (int i = 0; i < 3; ++i)
    {
        printf("t_proj_parameter[%d]: [%f, %f, %f %f; %f, %f, %f, %f; %f, %f, %f, %f;]\n", i,
            tex2D(t_proj_param_Nx12, 0, i), tex2D(t_proj_param_Nx12, 1, i), tex2D(t_proj_param_Nx12, 2, i),
            tex2D(t_proj_param_Nx12, 3, i), tex2D(t_proj_param_Nx12, 4, i), tex2D(t_proj_param_Nx12, 5, i),
            tex2D(t_proj_param_Nx12, 6, i), tex2D(t_proj_param_Nx12, 7, i), tex2D(t_proj_param_Nx12, 8, i),
            tex2D(t_proj_param_Nx12, 9, i), tex2D(t_proj_param_Nx12, 10, i), tex2D(t_proj_param_Nx12, 11, i)
            );
    }
}

'''

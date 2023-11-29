#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "cuda.h"
#include "cpu_bitmap.h"
#include <ctime>
#include <stdio.h>
#include <cmath>

#define MAX_DEPTH 5
#define DIM 800

#define SPHERESNUMBER 10
#define LIGHTSNUMBER 2
#define PI 3.14159265
const int fov = PI / 2.;

class Vec3f
{
public:
    float x, y, z;
    __device__ Vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    __device__ Vec3f operator-(const Vec3f& b) const { return Vec3f(x - b.x, y - b.y, z - b.z); }
    __device__ Vec3f operator-() const { return *this * (-1); }
    __device__ Vec3f operator+(const Vec3f& b) const { return Vec3f(x + b.x, y + b.y, z + b.z); }
    __device__ Vec3f operator*(double b) const { return Vec3f(x * b, y * b, z * b); }
    __device__ double operator*(const Vec3f& b) const { return x * b.x + y * b.y + z * b.z; }
    __device__ Vec3f cross(const Vec3f& b) const { return Vec3f(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
    __device__ double magnitude() const { return sqrt(x * x + y * y + z * z); }
    __device__  Vec3f normalize() const { return *this * (1 / magnitude()); }
};

struct Light
{
    __device__ Light(const Vec3f& position, const float& intensity) : position(position), intensity(intensity) {}
    Vec3f position;
    float intensity;
};

struct Material
{
    __device__ Material(const Vec3f& albedo, const Vec3f& diffuse_color, const float& specular) : albedo(albedo), diffuse_color(diffuse_color), specular(specular) {}
    __device__ Material() : albedo(1, 0, 0), diffuse_color(1.0, 0.0, 0.0), specular() {}
    Vec3f albedo;
    Vec3f diffuse_color;
    float specular;
};

struct Sphere
{
    Vec3f center;
    float radius;
    Material material;
    __device__ Sphere(const Vec3f& center, const float& radius, const Material& material) : center(center), radius(radius), material(material) {}
    __device__ bool RayIntersect(const Vec3f& ray_origin, const Vec3f& ray_direction, float& intersection_distance) const
    {
        Vec3f center_to_ray_origin = center - ray_origin;
        float projection_length = center_to_ray_origin * ray_direction;
        float squared_distance_from_center = center_to_ray_origin * center_to_ray_origin - projection_length * projection_length;
        if (squared_distance_from_center > radius * radius)
        {
            return false;
        }
        float half_circumference = sqrtf(radius * radius - squared_distance_from_center);
        intersection_distance = projection_length - half_circumference;
        float other_intersection_distance = projection_length + half_circumference;
        if (intersection_distance < 0)
        {
            intersection_distance = other_intersection_distance;
        }
        if (intersection_distance < 0)
        {
            return false;
        }
        return true;
    }
};

__device__ Vec3f Reflect(const Vec3f& directional, const Vec3f& normal)
{
    return directional - normal * 2.f * (directional * normal);
}

__device__ bool SceneIntersect(const Vec3f& orig, const Vec3f& dir, Sphere* spheres, Vec3f& hit_point, Vec3f& normal, Material& mat)
{
    float closest_sphere_dist = 1000000;
    for (int i = 0; i < SPHERESNUMBER; i++)
    {
        float dist_i;
        if (spheres[i].RayIntersect(orig, dir, dist_i) && dist_i < closest_sphere_dist)
        {
            closest_sphere_dist = dist_i;
            hit_point = orig + dir * dist_i;
            normal = (hit_point - spheres[i].center).normalize();
            mat = spheres[i].material;
        }
    }
    return closest_sphere_dist < 1000;
}

__device__ Vec3f CastRay(const Vec3f& original , const Vec3f& directional, Sphere* spheres, Light* lights, int depth)
{
    Vec3f point, N;
    Material material;

    if (depth > MAX_DEPTH || !SceneIntersect(original, directional, spheres, point, N, material))
    {
        //return Vec3f(51.0, 178.5, 204.0); // background color
        //return Vec3f(255.0, 255.0, 255.0); // background color
        return Vec3f(25.0, 25.0, 25.0); // background color
    }

    Vec3f reflect_directional = Reflect(directional, N).normalize();
    Vec3f reflect_original = reflect_directional * N < 0 ? point - N * 1e-3 : point + N * 1e-3; // offset the original point to avoid occlusion by the object itself
    Vec3f reflect_color = CastRay(reflect_original, reflect_directional, spheres, lights, depth + 1);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;

    for (int i = 0; i < LIGHTSNUMBER; i++)
    {
        Vec3f light_dir = (lights[i].position - point).normalize();
        float light_distance = (lights[i].position - point).magnitude();

        diffuse_light_intensity += lights[i].intensity * ((0.f > light_dir * N) ? 0.f : light_dir * N);
        specular_light_intensity += powf((0.f > -Reflect(-light_dir, N) * directional) ? 0.f : -Reflect(-light_dir, N) * directional, material.specular) * lights[i].intensity;
    }
    
    return material.diffuse_color * diffuse_light_intensity * material.albedo.x + Vec3f(1., 1., 1.) * specular_light_intensity * material.albedo.y + reflect_color * material.albedo.z;
}

__global__ void RayTracing(Sphere* spheres, Light* lights, unsigned char* image, int depth)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y; 
    int pixelIndex = x + y * blockDim.x * gridDim.x;

    float pixelX = (2 * (x + 0.5) / (float)DIM - 1) * tan(fov / 2.) * DIM / (float)DIM;
    float pixelY = -(2 * (y + 0.5) / (float)DIM - 1) * tan(fov / 2.);
    Vec3f rayDirection = Vec3f(pixelX, pixelY, 1).normalize();
    Vec3f rayColor = CastRay(Vec3f(0, 0, 0), rayDirection, spheres, lights, depth);

    image[pixelIndex * 4 + 0] = (int)(rayColor.x);
    image[pixelIndex * 4 + 1] = (int)(rayColor.y);
    image[pixelIndex * 4 + 2] = (int)(rayColor.z);
    image[pixelIndex * 4 + 3] = 255;
}

__global__ void GenerateSpheres(unsigned int seed, Sphere* spheres)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; 
    curandState_t state;

    if (index < SPHERESNUMBER)
    {
        curand_init(seed, index, 0, &state); // Use index as the offset to create different random seeds
        float randomMaterialIndex = curand_uniform(&state);
        Material sphereMaterial;

        if (randomMaterialIndex > 0.66)
        {
            sphereMaterial = Material(Vec3f(0.6, 0.3, 0.1), Vec3f(0.4, 0.4, 0.3), 50.);
        }
        else if (randomMaterialIndex < 0.33)
        {
            sphereMaterial = Material(Vec3f(0.0, 10.0, 0.8), Vec3f(1.0, 1.0, 1.0), 1425.);
        }
        else
        {
            sphereMaterial = Material(Vec3f(0.9, 0.1, 0.0), Vec3f(0.3, 0.1, 0.1), 10.);
        }

        spheres[index].material = sphereMaterial;
        spheres[index].center.x = (curand_uniform(&state) * 2 - 1) * DIM / 20; // Generate random value for x coordinate of center
        spheres[index].center.y = (curand_uniform(&state) * 2 - 1) * DIM / 20; // Generate random value for y coordinate of center
        spheres[index].center.z = 100 + curand_uniform(&state) * 50; // Generate random value for z coordinate of center
        spheres[index].radius = curand_uniform(&state) * DIM / 200 + 5; // Generate random value for radius
    }
}

__global__ void GenerateLights(unsigned int seed, Light* lights)
{
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t state;

    if (ind < LIGHTSNUMBER)
    {
        curand_init(seed, ind, 0, &state); // Use ind as the offset to create different random seeds
        lights[ind].intensity = 255;
        lights[ind].position.x = (2 * curand_uniform(&state) - 1) * DIM / 2; // Generate random value for x coordinate of light position
        lights[ind].position.y = (2 * curand_uniform(&state) - 1) * DIM / 2; // Generate random value for y coordinate of light position
        lights[ind].position.z = (2 * curand_uniform(&state) - 1) * DIM / 2; // Generate random value for z coordinate of light position
    }
}

struct DataBlock
{
    unsigned char* bitmap;
    Sphere* spheres;
};

int main()
{ 
    // Data block declaration.
    DataBlock data;

    // CUDA event declaration.
    cudaEvent_t start, stop;

    // CUDA event creation cudaEvent_t start, stop.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Recording the start time of the data processing operation.
    cudaEventRecord(start, 0);

    // Initialization of the scene and arrays of spheres and light sources.
    CPUBitmap bitmap(DIM, DIM, &data);
    unsigned char* dev_bitmap;
    Sphere* spheres;
    Light* lights;

    // Memory allocation.
    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
    cudaMalloc((void**)&spheres, sizeof(Sphere) * SPHERESNUMBER);
    cudaMalloc((void**)&lights, sizeof(Light) * LIGHTSNUMBER);

    // Initialization grids and threads.
    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    // Spheres generation.
    GenerateSpheres<<<1, SPHERESNUMBER>>> (time(0), spheres);
    
    // Wait for all the operations on the current device to finish.
    cudaDeviceSynchronize();
    
    // Lights generation.
    GenerateLights<<<1, LIGHTSNUMBER>>> (time(0), lights);
    
    // Wait for all the operations on the current device to finish.
    cudaDeviceSynchronize();

    // Display the parameters of spheres and light sources.
    Sphere* temp_spheres = (Sphere*)malloc(sizeof(Sphere) * SPHERESNUMBER);
    cudaMemcpy(temp_spheres, spheres, sizeof(Sphere) * SPHERESNUMBER, cudaMemcpyDeviceToHost);
    for (int i = 0; i < SPHERESNUMBER; i++)
    {
        printf("%f %f %f %f %f %f %f %f %f %f %f \n", temp_spheres[i].material.albedo.x, temp_spheres[i].material.albedo.y,
            temp_spheres[i].material.albedo.z, temp_spheres[i].material.diffuse_color.x, temp_spheres[i].material.diffuse_color.y,
            temp_spheres[i].material.diffuse_color.z, temp_spheres[i].material.specular, temp_spheres[i].center.x, temp_spheres[i].center.y, temp_spheres[i].center.z, temp_spheres[i].radius);
    }
    free(temp_spheres);
    
    printf("\n");

    Light* temp_lights = (Light*)malloc(sizeof(Light) * LIGHTSNUMBER);
    cudaMemcpy(temp_lights, lights, sizeof(Light) * LIGHTSNUMBER, cudaMemcpyDeviceToHost);
    for (int i = 0; i < LIGHTSNUMBER; i++)
    {
        printf("%f %f %f %f\n", temp_lights[i].intensity, temp_lights[i].position.x, temp_lights[i].position.y, temp_lights[i].position.z);
    }   
    free(temp_lights);
    
    // Ray tracing.
    RayTracing <<<grids, threads>>> (spheres, lights, dev_bitmap, 1);

    // Copy the image data from device to host.
    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

    // Record the end time of the data processing operation.
    cudaEventRecord(stop, 0);
    
    // Synchronize the start and stop events.
    cudaEventSynchronize(stop);

    // Elapsed time declaration.
    float elapsed_time;

    // Calculate the elapsed time of the data processing operation.
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Display elapsed time.
    printf("Time: %f", elapsed_time);

    // Destroy the start and stop events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Memory release.
    cudaFree(dev_bitmap);
    cudaFree(spheres);
    cudaFree(lights);

    // Save image as bmp file.
    bitmap.save_as_bmp("scene.bmp");

    // Display the image and exit.
    bitmap.display_and_exit();

    return 0;
}
/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix_world.h>
#include "commonStructs.h"
#include "helpers.h"

struct PerRayData_radiance
{
    float3 result; float importance;
    int depth;
    int mode;
    int mode_ret;
};

struct PerRayData_shadow
{
    float3 attenuation;
};

struct PerRayData_distance
{
    float distance;
};


rtDeclareVariable(int,               max_depth, , );
rtBuffer<BasicLight>                 lights;
rtDeclareVariable(float3,            ambient_light_color, , );
rtDeclareVariable(unsigned int,      radiance_ray_type, , );
rtDeclareVariable(unsigned int,      shadow_ray_type, , );

rtDeclareVariable(unsigned int,      distance_ray_type, , );

rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(rtObject,          top_shadower, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );

rtDeclareVariable(PerRayData_distance,   prd_distance, rtPayload, );

static __device__ void toonShadowed()
{
    // this material is opaque, so it fully attenuates all shadow rays
    prd_shadow.attenuation = optix::make_float3(0.0f);
    rtTerminateRay();
}

    static
__device__ float3 discretize(float3 color, float intensity)
{
    if (intensity > 0.95)
        color = make_float3(1.0,1.0,1.0) * color;
    else if (intensity > 0.5)
        color = make_float3(0.7,0.7,0.7) * color;
    else if (intensity > 0.05)
        color = make_float3(0.35,0.35,0.35) * color;
    else
        color = make_float3(0.1,0.1,0.1) * color;
    return color;
}

static
__device__ void edgeDetect() {
    prd.result = make_float3(1.0,1.0,1.0);
    prd.mode_ret = 1;
}

    static
__device__ void toonShade( float3 p_Kd,
        float3 p_Ka, //ambiance
        float3 p_Ks, //
        float3 p_Kr, //reflectance
        float  p_toon_exp, 
        float3 p_normal )
{
    float3 hit_point = ray.origin + t_hit * ray.direction;
    // ambient contribution
    float3 result = p_Ka * ambient_light_color;



    float intensity = 0.0;
    // compute direct lighting
    unsigned int num_lights = lights.size();
    for(int i = 0; i < num_lights; ++i) {
        BasicLight light = lights[i];
        float Ldist = optix::length(light.pos - hit_point);
        float3 L = optix::normalize(light.pos - hit_point);
        float nDl = optix::dot( p_normal, L);
        // cast shadow ray
        float3 light_attenuation = make_float3(static_cast<float>( nDl > 0.0f ));
        if ( nDl > 0.0f && light.casts_shadow ) {
            PerRayData_shadow shadow_prd;
            shadow_prd.attenuation = make_float3(1.0f);
            optix::Ray shadow_ray = optix::make_Ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
            rtTrace(top_shadower, shadow_ray, shadow_prd);
            light_attenuation = shadow_prd.attenuation;
        }

       

       //

        // If not completely shadowed, light the hit point
        if( fmaxf(light_attenuation) > 0.0f ) {
            float3 Lc = light.color * light_attenuation;
            result += p_Kd * nDl * Lc;
            intensity += nDl/num_lights;
        }
    }
    if( fmaxf( p_Kr ) > 0 ) {
        // ray tree attenuation
        PerRayData_radiance new_prd;             
        new_prd.importance = prd.importance * optix::luminance( p_Kr );
        new_prd.depth = prd.depth + 1;
        // reflection ray
        if( new_prd.importance >= 0.01f && new_prd.depth <= max_depth) {
            float3 R = optix::reflect( ray.direction, p_normal );
            optix::Ray refl_ray = optix::make_Ray( hit_point, R, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX );
            rtTrace(top_object, refl_ray, new_prd);
            result += p_Kr * new_prd.result;
        }
    }

    // START EDGE DETECT

    //unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame);


    //optix::Ray edge_ray = optix::make_Ray(hit_point-edge_test_dir*0.1f,
    //        edge_test_dir, distance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    //rtTrace(top_object, edge_ray, prd_distance);
    

    if(prd.depth < max_depth) {
        float3 edge_test_dir = ray.direction;
        edge_test_dir.x += 0.1f;
        edge_test_dir = optix::normalize(edge_test_dir);

        PerRayData_radiance new_prd;             
        new_prd.depth = max_depth+1;
        new_prd.mode = 1;
        new_prd.mode_ret = 0;

        optix::Ray edge_ray = optix::make_Ray(hit_point-ray.direction*1.0f,
                edge_test_dir, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

        rtTrace(top_object, edge_ray, new_prd);
        if (new_prd.mode_ret == 1) {
            result += new_prd.result;
        }
    }

    //rtTrace(
    // pass the color back up the tree
    prd.result = discretize( result, intensity );
}

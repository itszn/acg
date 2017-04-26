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

using namespace optix;

rtDeclareVariable(float3,  x, , );
rtDeclareVariable(float3,  y, , );
rtDeclareVariable(float3,  z, , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


RT_PROGRAM void robust_intersect(int primIdx)
{
    // check if we intersect the plane of the triangle
    float3 n = normalize(cross(y - x, z - x));
    float d = dot(n, x);

    float t = (d - dot(n, ray.origin)) / dot(n, ray.direction);
    if (t < 0.00001)
        return;
    // check that point is within triangle
    float3 p = ray.direction * t + ray.origin;
    float3 tmp[3] = { x, y, z };
    for (int i = 0; i < 3; ++i) {
        float3 v1 = tmp[i] - p;
        float3 v2 = tmp[(i+1) % 3] - p;
        float3 n = normalize(cross(v2, v1));
        float d = dot(-ray.origin, n);
        if (dot(p,n) + d < 0)
            return;
    }
    if (rtPotentialIntersection(t)) {
        geometric_normal = shading_normal = n;
        rtReportIntersection(0);
    }
}


RT_PROGRAM void bounds (int, float result[6])
{
    result[0] = x.x;
    result[0] = min(result[0], y.x);
    result[0] = min(result[0], z.x);

    result[1] = x.x;
    result[1] = max(result[1], y.x);
    result[1] = max(result[1], z.x);

    result[2] = x.y;
    result[2] = min(result[2], y.y);
    result[2] = min(result[2], z.y);

    result[3] = x.y;
    result[3] = max(result[3], y.y);
    result[3] = max(result[3], z.y);

    result[4] = x.z;
    result[4] = min(result[4], y.z);
    result[4] = min(result[4], z.z);

    result[5] = x.z;
    result[5] = max(result[5], y.z);
    result[5] = max(result[5], z.z);
}

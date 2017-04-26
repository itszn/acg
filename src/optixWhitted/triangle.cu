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
    rtPotentialIntersection(t);

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
    rtReportIntersection(0);
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

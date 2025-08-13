// light_renderer.slang

struct Ray {
    float3 origin;
    float3 dir;
};

struct Hit {
    bool valid;
    float t;
    float3 position;
    float3 normal;
};

struct Mesh {
    float3 vertices[];
    int3 triangles[];
};

// Generate a ray in world space from camera uv and pose
Ray generateRay(float2 uv, float4x4 camPose) {
    float3 camOrigin = camPose[3].xyz;
    float3 dir = normalize(float3(uv - 0.5, 1.0)); // Simplified camera model
    dir = mul((float3x3)camPose, dir);
    return Ray(camOrigin, dir);
}

// Moller-Trumbore ray-triangle intersection
Hit intersectRayMesh(Ray ray, Mesh mesh) {
    Hit bestHit;
    bestHit.valid = false;
    bestHit.t = 1e30;

    for (int i = 0; i < mesh.triangles.length; ++i) {
        int3 tri = mesh.triangles[i];
        float3 v0 = mesh.vertices[tri.x];
        float3 v1 = mesh.vertices[tri.y];
        float3 v2 = mesh.vertices[tri.z];

        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        float3 h = cross(ray.dir, edge2);
        float a = dot(edge1, h);
        if (abs(a) < 1e-5) continue;

        float f = 1.0 / a;
        float3 s = ray.origin - v0;
        float u = f * dot(s, h);
        if (u < 0.0 || u > 1.0) continue;

        float3 q = cross(s, edge1);
        float v = f * dot(ray.dir, q);
        if (v < 0.0 || u + v > 1.0) continue;

        float t = f * dot(edge2, q);
        if (t > 1e-3 && t < bestHit.t) {
            bestHit.valid = true;
            bestHit.t = t;
            bestHit.position = ray.origin + t * ray.dir;
            bestHit.normal = normalize(cross(edge1, edge2));
        }
    }
    return bestHit;
}

float renderScene(
    float2 uv,
    float4x4 camPose,
    Mesh mesh,
    float3 lightPos,
    float intensity
) {
    Ray ray = generateRay(uv, camPose);
    Hit hit = intersectRayMesh(ray, mesh);
    if (!hit.valid) return 0.0;

    float3 L = normalize(lightPos - hit.position);
    float NdotL = max(dot(hit.normal, L), 0.0);
    float dist2 = dot(lightPos - hit.position, lightPos - hit.position);
    return intensity * NdotL / (dist2 + 1e-3);
}

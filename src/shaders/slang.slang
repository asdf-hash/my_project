struct InstanceData
{
    float3 worldPos;
    uint colorIndex;
};

struct UniformBuffer
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

StructuredBuffer<InstanceData> instanceData;
ConstantBuffer<UniformBuffer> ubo;

struct MeshVertex
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
    uint colorIndex : TEXCOORD1;
};

static const float2 QuadOffsets[4] = {
    float2(0.0, 0.0),
    float2(1.0, 0.0),
    float2(1.0, 1.0),
    float2(0.0, 1.0)
};
static const float3 Normals[6] = {
    float3(1.0, 0.0, 0.0),
    float3(-1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, -1.0, 0.0),
    float3(0.0, 0.0, 1.0),
    float3(0.0, 0.0, -1.0),
};
static const float3 ColorLUT[2] = {
    float3(1.0, 1.0, 1.0),
    float3(0.133, 0.545, 0.133),
};

Texture2D texture;
SamplerState sampler;

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshMain(
    out vertices MeshVertex verts[4],
    out indices uint3 triangles[2],
    uint3 dispatchThreadID: SV_DispatchThreadID)
{
    SetMeshOutputCounts(4, 2);

    uint instanceIndex = dispatchThreadID.x;
    InstanceData instance = instanceData[instanceIndex];

    [unroll]
    for (uint i = 0; i < 4; ++i)
    {
        float3 localPos = float3(QuadOffsets[i], 0.0);
        float3 worldPos = instance.worldPos + localPos;
        verts[i].pos = mul(ubo.proj, mul(ubo.view, float4(worldPos, 1.0)));
        verts[i].uv = QuadOffsets[i];
        verts[i].colorIndex = instance.colorIndex;
    }

    triangles[0] = uint3(0, 1, 2);
    triangles[1] = uint3(0, 2, 3);
}

struct FragmentInput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
    uint colorIndex : TEXCOORD1;
};

[shader("fragment")]
float4 fragMain(FragmentInput input) : SV_TARGET
{
    float3 tint = ColorLUT[min(input.colorIndex, 1u)];
    return texture.Sample(sampler, input.uv) * float4(tint, 1.0);
}

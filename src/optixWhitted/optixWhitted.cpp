#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#  include <GL/wglew.h>
#  include <GL/freeglut.h>
#  else
#  include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "commonStructs.h"

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <tuple>

using namespace optix;

const uint32_t width  = 768u;
const uint32_t height = 768u;

auto ptxPath(const std::string& cuda_file) -> std::string
{
    return std::string(sutil::samplesPTXDir()) +
            "/optixWhitted_generated_" +
            cuda_file +
            ".ptx";
}

auto parse_obj_file(std::string path) -> std::vector<std::tuple<float, float, float>>
{
    std::vector<std::tuple<float, float, float>> triangles;
    return triangles;
}

auto create_context() -> Context
{
    Context context = Context::create();
    context->setRayTypeCount(2);
    context->setEntryPointCount(1);
    context->setStackSize(2800);

    context["max_depth"]->setInt( 10 );
    context["radiance_ray_type"]->setUint( 0 );
    context["shadow_ray_type"]->setUint( 1 );
    context["distance_ray_type"]->setUint( 2 );
    context["frame"]->setUint( 0u );
    context["scene_epsilon"]->setFloat( 1.e-4f );
    context["ambient_light_color"]->setFloat( 0.4f, 0.4f, 0.4f );

    // construct png output buffer
    Buffer output = sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4,
                                              width, height, false);
    Buffer accum  = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                          RT_FORMAT_FLOAT4, width, height);
    context["output_buffer"]->set(output);
    context["accum_buffer"]->set(accum);

    // Ray generation program
    std::string ptx_path(ptxPath("accum_camera.cu"));
    Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "pinhole_camera");
    context->setRayGenerationProgram(0, ray_gen_program);

    // Exception program
    Program exception_program = context->createProgramFromPTXFile(ptx_path, "exception");
    context->setExceptionProgram(0, exception_program);
    context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

    // Miss program
    ptx_path = ptxPath("constantbg.cu");
    context->setMissProgram(0, context->createProgramFromPTXFile(ptx_path, "miss"));
    context["bg_color"]->setFloat(0.34f, 0.55f, 0.85f);

    return context;
}

auto createScene(Context &context) -> GeometryInstance
{
    // setup floor
    auto floor_ptx = ptxPath("parallelogram.cu");
    auto parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount(1u);
    parallelogram->setBoundingBoxProgram(
            context->createProgramFromPTXFile(floor_ptx, "bounds"));
    parallelogram->setIntersectionProgram(
            context->createProgramFromPTXFile(floor_ptx, "intersect"));
    float3 anchor = make_float3( -16.0f, 0.01f, -8.0f );
    float3 v1 = make_float3( 32.0f, 0.0f, 0.0f );
    float3 v2 = make_float3( 0.0f, 0.0f, 16.0f );
    float3 normal = cross( v1, v2 );
    normal = normalize( normal );
    float d = dot( normal, anchor );
    v1 *= 1.0f/dot( v1, v1 );
    v2 *= 1.0f/dot( v2, v2 );
    float4 plane = make_float4( normal, d );
    parallelogram["plane"]->setFloat( plane );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );
    parallelogram["anchor"]->setFloat( anchor );

    // Checker material for floor
    const std::string checker_ptx = ptxPath( "checker.cu" );
    Program check_ch = context->createProgramFromPTXFile( checker_ptx, "closest_hit_radiance" );
    Program check_ah = context->createProgramFromPTXFile( checker_ptx, "any_hit_shadow" );
    Material floor_matl = context->createMaterial();
    floor_matl->setClosestHitProgram( 0, check_ch );
    floor_matl->setAnyHitProgram( 1, check_ah );

    floor_matl["Kd1"]->setFloat( 0.8f, 0.3f, 0.15f);
    floor_matl["Ka1"]->setFloat( 0.8f, 0.3f, 0.15f);
    floor_matl["Ks1"]->setFloat( 0.0f, 0.0f, 0.0f);
    floor_matl["Kd2"]->setFloat( 0.9f, 0.85f, 0.05f);
    floor_matl["Ka2"]->setFloat( 0.9f, 0.85f, 0.05f);
    floor_matl["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f);
    floor_matl["inv_checker_size"]->setFloat( 32.0f, 16.0f, 1.0f );
    floor_matl["toon_exp1"]->setFloat( 0.0f );
    floor_matl["toon_exp2"]->setFloat( 0.0f );
    floor_matl["Kr1"]->setFloat( 0.0f, 0.0f, 0.0f);
    floor_matl["Kr2"]->setFloat( 0.0f, 0.0f, 0.0f);

    return context->createGeometryInstance(
            parallelogram, &floor_matl, &floor_matl + 1);
}

void setup_lights(Context &context)
{
    BasicLight lights[] = {
        { make_float3( 60.0f, 40.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
    };

    Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(BasicLight));
    light_buffer->setSize(sizeof(lights)/sizeof(lights[0]));
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    context["lights"]->set(light_buffer);
}

void setup_camera(Context &context)
{
    const float vfov  = 60.0f;
    const float aspect_ratio = static_cast<float>(width) /
        static_cast<float>(height);

    float3 camera_up          = make_float3(8.0f, 2.0f, -4.0f);
    float3 camera_lookat      = make_float3(4.0f, 2.3f, -4.0f);
    float3 camera_eye         = make_float3(0.0f, 1.0f,  0.0f);
    Matrix4x4 camera_rotate   = Matrix4x4::identity();

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up,
            vfov, aspect_ratio, camera_u, camera_v, camera_w, true);

    const Matrix4x4 frame = Matrix4x4::fromBasis(
            normalize( camera_u),
            normalize( camera_v),
            normalize(-camera_w),
            camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans   = frame*camera_rotate*camera_rotate*frame_inv;

    camera_eye    = make_float3(trans*make_float4(camera_eye, 1.0f));
    camera_lookat = make_float3(trans*make_float4(camera_lookat, 1.0f));
    camera_up     = make_float3(trans*make_float4(camera_up, 0.0f));

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up,
            vfov, aspect_ratio, camera_u, camera_v, camera_w, true);
    camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat(camera_eye);
    context["U"  ]->setFloat(camera_u);
    context["V"  ]->setFloat(camera_v);
    context["W"  ]->setFloat(camera_w);
}

void glutInitialize(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Toshi's Reverse Shell");
    glutHideWindow();
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << argv[0] << " <input_obj> <output_file>" << std::endl; 
        return EXIT_FAILURE;
    }
    std::string input_obj = argv[1];
    std::string out_file  = argv[2];
    auto triangles = parse_obj_file(std::move(input_obj));

    glutInitialize(&argc, argv);
    glewInit();
    auto context = create_context();

    // Create GIs for each piece of geometry
    std::vector<GeometryInstance> gis;
    gis.push_back(createScene(context));

    // Place all in group
    GeometryGroup geometrygroup = context->createGeometryGroup();
    geometrygroup->setChildCount(static_cast<unsigned int>(gis.size()));
    for (unsigned int i = 0; i < gis.size(); ++i)
        geometrygroup->setChild(i, gis[i]);
    geometrygroup->setAcceleration( context->createAcceleration("NoAccel") );
    context["top_object"]->set(geometrygroup);
    context["top_shadower"]->set(geometrygroup);

    // setup lights, camera
    setup_lights(context);
    setup_camera(context);

    context->validate();
    context->launch(0, width, height);
    sutil::displayBufferPPM(out_file.c_str(),
            context["output_buffer"]->getBuffer());
    context->destroy();
    return EXIT_SUCCESS;
}

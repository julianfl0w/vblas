#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#extension GL_ARB_separate_shader_objects : enable
DEFINE_STRING// This will be (or has been) replaced by constant definitions
    
BUFFERS_STRING// This will be (or has been) replaced by buffer definitions
    
layout (local_size_x = THREADS_PER_WORKGROUP, local_size_y = 1, local_size_z = 1 ) in;

void main() {
    uint workgroup_ix = gl_GlobalInvocationID.x;
    VARIABLEDECLARATIONS
    sumOut[workgroup_ix] = x[workgroup_ix] OPERATION y[workgroup_ix%YLEN];
}
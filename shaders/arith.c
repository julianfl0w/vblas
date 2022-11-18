#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#extension GL_ARB_separate_shader_objects : enable
DEFINE_STRING// This will be (or has been) replaced by constant definitions
    
BUFFERS_STRING// This will be (or has been) replaced by buffer definitions
    
layout (local_size_x = THREADS_PER_WORKGROUP, local_size_y = 1, local_size_z = 1 ) in;

void main() {
    uint WORKGROUP_COUNT   = gl_NumWorkGroups.x;
    //uint workgroup_ix = gl_GlobalInvocationID.x;
    uint workgroup_ix = gl_WorkGroupID.x;
    uint thread_ix    = gl_LocalInvocationID.x;
    uint shader_ix    = workgroup_ix*OPS_PER_THREAD;
    uint workgroupStart_ix = workgroup_ix*OPS_PER_THREAD*THREADS_PER_WORKGROUP;
    
    VARIABLEDECLARATIONS
        
    //for(uint i = shader_ix; i < shader_ix + OPS_PER_THREAD; i++)
    //    sumOut[i] = x[i] OPERATION y[i%YLEN];
    for (uint i = thread_ix; i < THREADS_PER_WORKGROUP*OPS_PER_THREAD; i += THREADS_PER_WORKGROUP){
        uint j = workgroupStart_ix + i;
        sumOut[j] = x[j] OPERATION y[j%YLEN];//];
    }
}
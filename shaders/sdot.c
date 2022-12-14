#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_scalar_block_layout : require
DEFINE_STRING // This will be (or has been) replaced by constant definitions
layout (local_size_x = LOCAL_X, local_size_y = LOCAL_Y, local_size_z = LOCAL_Z ) in;
BUFFERS_STRING  // This will be (or has been) replaced by buffer definitions
    
//float64_t vectordot(in float64_t* a, in float64_t* b, in uint length){
//    sum = 0;
//    for(uint i = 0; i < length; i++){
//        sum += a[i] * b[i];
//    }
//    
//    return sum;
//}

void main() {
    //precision mediump float;
    //precision mediump float64_t;

    //uint sampleNo = gl_LocalInvocationID.x;
    //uint shaderIndexInSample = gl_LocalInvocationID.y;
    //uint zindex = gl_LocalInvocationID.z;
    VARIABLEDECLARATIONS
    
    //uint globalIndex = gl_LocalInvocationIndex;
    //uint workgroupNo    = gl_GlobalInvocationID;
    uint workgroupNo    = gl_WorkGroupID.x;
    uint localSize      = LOCAL_X*LOCAL_Y*LOCAL_Z;
    uint workgroupCount = gl_NumWorkGroups.x;
    uint localIndex     = gl_LocalInvocationID.x*LOCAL_Y + gl_LocalInvocationID.y;
    uint globalIndex    = workgroupNo*localSize + localIndex;
    uint globalSize     = localSize*workgroupCount;
    
    // assume workgroup size 1
    //uint globalIndex      = workgroupNo*workgroupSize + indexInWorkgroup;
    
    // now break the problem into workgroupSize parts 
    // (for max compatibility, 32x16 = 512
    
    uint wPerShader = uint(XDIM0 / globalSize) + 1;
    
    PROCTYPE sum = 0;
    // assume X array is 2 dimensions, Y array is 1d
    // x array, b dimension is the same as y array, 
    uint startW = globalIndex*wPerShader;
    uint endW   = (globalIndex+1)*wPerShader;
    
    for(uint w = startW; w < endW; w++){
        //Z[w] = vectordot(X+w*XDIM1, Y, XDIM1);
        
        if(w >= XDIM0)
            break;
        uint startN = w*XDIM1;
        sum = 0;
        
        for(uint n = 0; n < XDIM1; n++){
            uint nx =startN + n;
            PROCTYPE x;
            x = X[nx];
            //x = sin(n);
            //x = n;
            thisAdd = x * Y[n];
            sum += thisAdd;
        }
        
        //sum = dot(X, Y);

        Z[w] = sum;
    }
    
    //for(uint i = 0; i < 10; i++)
    //    if(i == globalIndex)
    //        Z[i] = i + 1;
}

// Multiple W example
// (ex. for Loiacono Transform)
// [ar0, br0, cr0, dr0 ... zr0] * [as0, bs0, cs0 ... zs0] 
// [ar1, br1, cr1, dr1 ... zr1]                   
// [..., ..., ..., ... ... ...]                   
// [arw, brw, crw, drw ... zrw]                   
//
// each dispatch should handle 1 (or more) w rows
// in the case of Twelve-Tone Equal Temperment, 
// provide 128 notes (MIDI STANDARD)
// this allows for 512/128 = 4 near-frequencies for each note
//. W is on dimension 0
// Provide X is the EIWN array, Y is the real signal

// Complex numbers example
// (ex. for Loiacono Transform)
// [ar0, br, cr, dr ... zr]
// [ai0, bi, ci, di ... zi] 
//
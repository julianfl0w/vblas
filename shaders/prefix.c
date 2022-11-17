// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// A prefix sum.
//
// This test builds in three configurations. The default is a
// compatibility mode, essentially plain GLSL. With ATOMIC set, the
// flag loads and stores are atomic operations, but uses barriers.
// With both ATOMIC and VKMM set, it uses acquire/release semantics
// instead of barriers.

#version 450

#extension GL_KHR_memory_scope_semantics : enable
#define ATOMIC 

#ifdef VKMM
#pragma use_vulkan_memory_model
#define ACQUIRE gl_StorageSemanticsBuffer, gl_SemanticsAcquire
#define RELEASE gl_StorageSemanticsBuffer, gl_SemanticsRelease
#else
#define ACQUIRE 0, 0
#define RELEASE 0, 0
#endif

#define OPS_PER_LOCALGROUP (THREADS_PER_LOCALGROUP * OPS_PER_Thread)

DEFINE_STRING // This will be (or has been) replaced by constant definitions
layout (local_size_x = LOCAL_X, local_size_y = LOCAL_Y, local_size_z = LOCAL_Z ) in;
BUFFERS_STRING  // This will be (or has been) replaced by buffer definitions
    
// These correspond to X, A, P respectively in the prefix sum paper.
#define FLAG_NOT_READY 0u
#define FLAG_AGGREGATE_READY 1u
#define FLAG_PREFIX_READY 2u


shared PROCTYPE sh_threadAggrigate[THREADS_PER_LOCALGROUP];


shared uint sh_localGroup_ix;
shared PROCTYPE sh_prefix;
shared uint sh_flag;

PROCTYPE salientOperation(PROCTYPE a, PROCTYPE b) {
    return (a + b);
}

// assume XSIZE = 512, YSIZE = ZSIZE = 1
void main() {
    uint localGroup_ix = gl_WorkGroupID.x;
    uint thread_ix     = gl_LocalInvocationID.x;
    
    VARIABLEDECLARATIONS
    PROCTYPE threadAggrigate[OPS_PER_Thread];
    
    /*
    // Determine localgroup to process by atomic counter (described in Section
    // 4.4 of prefix sum paper).
    if (thread_ix == 0) {
        sh_localGroup_ix = atomicAdd(part_counter[0], 1);
    }
    barrier();
    uint localGroup_ix = sh_localGroup_ix;*/
    
    uint start_global_ix = localGroup_ix * OPS_PER_LOCALGROUP + thread_ix * OPS_PER_Thread;

    // 4.1.1, 4.1.2 unnecessary
    
    // 4.1.3
    //Compute and record the partition-wide aggregate. Each
    //processor computes and records its partition-wide
    //aggregate to the corresponding partition descriptor.
    
    threadAggrigate[0] = inbuf[start_global_ix];
    
    // do the Thread work here
    for (uint i = 1; i < OPS_PER_Thread; i++) {
        threadAggrigate[i] = salientOperation(threadAggrigate[i - 1], inbuf[start_global_ix + i]);
    }
    // save it to shared memory
    PROCTYPE thisThreadAgg = threadAggrigate[OPS_PER_Thread - 1];
    sh_threadAggrigate[thread_ix] = thisThreadAgg;
    
    // calculate this thread's inclusive prefix by looking 
    // progressively backwards at aggrigates from previous threads
    barrier();
    memoryBarrierBuffer();
    
    
    // It then executes a memory fence and updates the descriptor’s
    // status_flag to A. Furthermore, the processor owning the
    // first partition copies aggregate to the inclusive_prefix
    // field, updates status_flag to P, and skips to Step 6 below.
    
    localGroup_flag[
    barrier();
    memoryBarrierBuffer();
    
    // Publish aggregate for this localgroup
    if (thread_ix == THREADS_PER_LOCALGROUP - 1) {
        aggregate[localGroup_ix] = thisThreadAgg;
        if (localGroup_ix == 0) {
            prefix[workGroup_ix] = thisThreadAgg;
        }
    }
    
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        barrier();
        if (thread_ix >= (1u << i)) {
            PROCTYPE other = sh_threadAggrigate[thread_ix - (1u << i)];
            thisThreadAgg = other + thisThreadAgg;
        }
        barrier();
        sh_threadAggrigate[thread_ix] = thisThreadAgg;
    }

    // Write flag with release semantics; this is done portably with a barrier.
    memoryBarrierBuffer();
    if (thread_ix == THREADS_PER_LOCALGROUP - 1) {
        uint thisflag = FLAG_AGGREGATE_READY;
        if (localGroup_ix == 0) {
            thisflag = FLAG_PREFIX_READY;
        }
#ifdef ATOMIC
        atomicStore(flag[localGroup_ix], thisflag, gl_ScopeDevice, RELEASE);
#else
        flag[localGroup_ix] = thisflag;
#endif
    }

}

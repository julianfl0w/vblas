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

#define OPS_PER_WORKGROUP (THREADS_PER_WORKGROUP * OPS_PER_THREAD)

DEFINE_STRING // This will be (or has been) replaced by constant definitions
layout (local_size_x = LOCAL_X, local_size_y = LOCAL_Y, local_size_z = LOCAL_Z ) in;
BUFFERS_STRING  // This will be (or has been) replaced by buffer definitions
    
// These correspond to X, A, P respectively in the prefix sum paper.
#define FLAG_NOT_READY 0u
#define FLAG_AGGREGATE_READY 1u
#define FLAG_PREFIX_READY 2u


shared PROCTYPE sh_threadAggrigate[THREADS_PER_WORKGROUP];
shared PROCTYPE sh_threadInclusivePrefix[THREADS_PER_WORKGROUP];


shared uint sh_workGroup_ix;
shared PROCTYPE sh_prefix;
shared uint sh_flag;

PROCTYPE salientOperation(PROCTYPE a, PROCTYPE b) {
    return (a + b);
}

// assume XSIZE = 512, YSIZE = ZSIZE = 1
void main() {
    uint workGroup_ix  = gl_WorkGroupID.x;
    uint thread_ix     = gl_LocalInvocationID.x;
    
    VARIABLEDECLARATIONS
    PROCTYPE threadAggrigate[OPS_PER_THREAD];
    
    uint WORKGROUP_COUNT   = gl_NumWorkGroups.x;
    uint workGroup_ix      = gl_WorkGroupID.x;
    uint thread_ix         = gl_LocalInvocationID.x;
    uint shader_ix         = workGroup_ix*OPS_PER_THREAD;
    uint workgroupStart_ix = workGroup_ix*OPS_PER_THREAD*THREADS_PER_WORKGROUP;
    uint thread_start_ix   = workgroupStart_ix + thread_ix;

    // My own take on Decoupled Lookback Prefix Scan
    // https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
    
    // 1: compute the prefix scan for this thread, store prefix in local memory
    // 2: copy the final value to shared memory. execute a barrier and memoryBarrierBuffer
    // 3: look progressively backwards at other threads. 
    //     a) while aggrigate unavailable, spin
    //     b) if aggrigate available, add to our prefix, look back one further. if at 0, we have our prefix and done
    //     c) if prefix available, add to our values, then done (unless this is thread 0)
    
    // do the Thread work here
    threadAggrigate[0] = inbuf[thread_start_ix];
    for (uint i = 1; i < OPS_PER_THREAD; i++) {
        // skip THREADS_PER_WORKGROUP to keep threads coalesced
        uint readIndex = thread_start_ix + i*THREADS_PER_WORKGROUP;
        threadAggrigate[i] = salientOperation(threadAggrigate[i - 1], inbuf[readIndex]);
    }
    
    // save aggrigate to shared memory
    // if this is not 0th thread
    if(thread_ix != 0){
        PROCTYPE thisThreadAgg = threadAggrigate[OPS_PER_THREAD - 1];
        atomicStore(sh_threadAggrigate[thread_ix], thisThreadAgg, gl_ScopeDevice, RELEASE);
        // set our flag to AGGRIGATE READY
        atomicStore(sh_flag[thread_ix], FLAG_AGGREGATE_READY, gl_ScopeDevice, RELEASE);
        
        // calculate this thread's inclusive prefix by looking 
        // progressively backwards at aggrigates from previous threads
        PROCTYPE thisThreadInclusivePrefix = thisThreadAgg;
        for(uint previousThread = thread_ix-1; previousThread >= 0; previousThread--){
            // wait for something to be ready
            uint previousThreadFlag
            while(1){
                previousThreadFlag = atomicLoad(sh_flag[previousThread], gl_ScopeDevice, ACQUIRE);
                if(previousThreadFlag != FLAG_NOT_READY)
                    break;
            }
            // preferred case. prefix is ready, we add it to our prefix and stop
            if(previousThreadFlag == FLAG_PREFIX_READY){
                thisThreadInclusivePrefix += atomicLoad(sh_threadInclusivePrefix[previousThread], gl_ScopeDevice, ACQUIRE);
                break;
            }
            // also a nice case. previous aggrigate is ready, we add it to our prefix and go back again
            else{ // FLAG_AGGREGATE_READY
                thisThreadInclusivePrefix += atomicLoad(sh_threadAggrigate[previousThread], gl_ScopeDevice, ACQUIRE);
            }
        }
    }
    
    
    // save prefix to shared memory
    atomicStore(sh_threadAggrigate[thread_ix], thisThreadInclusivePrefix, gl_ScopeDevice, RELEASE);
    // set our flag to PREFIX READY
    atomicStore(sh_flag[thread_ix], FLAG_PREFIX_READY, gl_ScopeDevice, RELEASE);
        
    // at this point, we have 
    // a) the prefix scan internal to this thread, 
    // b) the prefix to this thread internal to this workgroup
    // we just need c) the prefix to this workgroup
    
    // Now only the final thread is active. 
    if (thread_ix == THREADS_PER_WORKGROUP - 1) {
        //Publish aggregate for this WORKGROUP
        atomicStore(aggregate[workGroup_ix], thisThreadInclusivePrefix, gl_ScopeDevice, RELEASE);
        // set our flag to AGGRIGATE READY
        atomicStore(flag[workGroup_ix], FLAG_AGGREGATE_READY, gl_ScopeDevice, RELEASE);
        
        // start looking at previous workgroup aggrigates
        // this might look familiar
        // save aggrigate to shared memory
        // if this is not 0th thread
        if(workGroup_ix != 0){
            PROCTYPE thisThreadAgg = threadAggrigate[OPS_PER_THREAD - 1];
            atomicStore(sh_threadAggrigate[thread_ix], thisThreadAgg, gl_ScopeDevice, RELEASE);
            // set our flag to AGGRIGATE READY
            atomicStore(sh_flag[thread_ix], FLAG_AGGREGATE_READY, gl_ScopeDevice, RELEASE);

            // calculate this thread's inclusive prefix by looking 
            // progressively backwards at aggrigates from previous threads
            PROCTYPE thisThreadInclusivePrefix = thisThreadAgg;
            for(uint previousThread = thread_ix-1; previousThread >= 0; previousThread--){
                // wait for something to be ready
                uint previousThreadFlag
                while(1){
                    previousThreadFlag = atomicLoad(sh_flag[previousThread], gl_ScopeDevice, ACQUIRE);
                    if(previousThreadFlag != FLAG_NOT_READY)
                        break;
                }
                // preferred case. prefix is ready, we add it to our prefix and stop
                if(previousThreadFlag == FLAG_PREFIX_READY){
                    thisThreadInclusivePrefix += atomicLoad(sh_threadInclusivePrefix[previousThread], gl_ScopeDevice, ACQUIRE);
                    break;
                }
                // also a nice case. previous aggrigate is ready, we add it to our prefix and go back again
                else{ // FLAG_AGGREGATE_READY
                    thisThreadInclusivePrefix += atomicLoad(sh_threadAggrigate[previousThread], gl_ScopeDevice, ACQUIRE);
                }
            }
        }
        
        // save prefix to shared memory
        atomicStore(sh_threadAggrigate[thread_ix], thisThreadInclusivePrefix, gl_ScopeDevice, RELEASE);
        // set our flag to PREFIX READY
        atomicStore(sh_flag[thread_ix], FLAG_PREFIX_READY, gl_ScopeDevice, RELEASE);

    }
}

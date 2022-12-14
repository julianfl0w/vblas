// From https://github.com/linebender/piet-gpu/blob/prefix/piet-gpu-hal/examples/shader/prefix.comp
// See https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf

#version 450
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_memory_scope_semantics : enable
// #extension VK_EXT_shader_atomic_float : require NOT WORKING

// One workgroup processes workgroup size * ELEMENTS_PER_WORKGROUP elements.
#define ELEMENTS_PER_WORKGROUP 16

DEFINE_STRING// This will be (or has been) replaced by constant definitions
BUFFERS_STRING// This will be (or has been) replaced by buffer definitions
    

// work_buf[0] is the tile id
// work_buf[i * 4 + 1] is the flag for tile i
// work_buf[i * 4 + 2] is the aggregate for tile i
// work_buf[i * 4 + 3] is the prefix for tile i


// These correspond to X, A, P respectively in the paper.
#define FLAG_NOT_READY 0
#define FLAG_AGGREGATE_READY 1
#define FLAG_PREFIX_READY 2

shared uint shared_tile;
shared uint shared_prefix;
// Note: the subgroup size and other dimensions are hard-coded.
// TODO: make it more adaptive.
shared PROCTYPE chunks[32];

void main() {
    VARIABLEDECLARATIONS
    uint local_ix = gl_LocalInvocationID.x;
    // Determine tile to process by atomic counter (implement idea from
    // section 4.4 in the paper).
    if (local_ix == 0) {
        shared_tile = atomicAdd(work_buf[0], 1);
    }
    barrier();
    uint my_tile = shared_tile;
    uint mem_base = my_tile * 16384;
    PROCTYPE aggregates[ELEMENTS_PER_WORKGROUP];

    // Interleave reading of data, computing row prefix sums, and aggregate
    // (step 3 of paper).
    PROCTYPE total = 0;
    for (uint i = 0; i < ELEMENTS_PER_WORKGROUP; i++) {
        uint ix = (local_ix & 0x3e0) * ELEMENTS_PER_WORKGROUP + i * 32 + (local_ix & 0x1f);
        PROCTYPE data = inbuf[mem_base + ix];
        PROCTYPE row = subgroupInclusiveAdd(data);
        total += row;
        aggregates[i] = row;
    }
    if (gl_SubgroupInvocationID == 31) {
        chunks[local_ix >> 5] = total;
    }

    barrier();
    if (local_ix < 32) {
        PROCTYPE chunk = chunks[gl_SubgroupInvocationID];
        total = subgroupInclusiveAdd(chunk);
        chunks[gl_SubgroupInvocationID] = total;
    }

    uint exclusive_prefix = 0;
    if (local_ix == 31) {
        atomicStore(work_buf[my_tile * 4 + 2], uint(total*CAST_PRECISION), gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
        uint flag = FLAG_AGGREGATE_READY;
        if (my_tile == 0) {
            atomicStore(work_buf[my_tile * 4 + 3], uint(total*CAST_PRECISION), gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
            flag = FLAG_PREFIX_READY;
        }
        atomicStore(work_buf[my_tile * 4 + 1], flag, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
        if (my_tile != 0) {
            // step 4: decoupled lookback
            uint look_back_ix = my_tile - 1;
            while (true) {
                flag = atomicLoad(work_buf[look_back_ix * 4 + 1], gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
                if (flag == FLAG_PREFIX_READY) {
                    uint their_prefix = atomicLoad(work_buf[look_back_ix * 4 + 3], gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
                    exclusive_prefix = their_prefix + exclusive_prefix;
                    break;
                } else if (flag == FLAG_AGGREGATE_READY) {
                    uint their_agg = atomicLoad(work_buf[look_back_ix * 4 + 2], gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
                    exclusive_prefix = their_agg + exclusive_prefix;
                    look_back_ix--;
                }
                // else spin
            }

            // step 5: compute inclusive prefix
            PROCTYPE inclusive_prefix = exclusive_prefix + total;
            shared_prefix = exclusive_prefix;
            atomicStore(work_buf[my_tile * 4 + 3], uint(inclusive_prefix*CAST_PRECISION), gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
            flag = FLAG_PREFIX_READY;
            atomicStore(work_buf[my_tile * 4 + 1], flag, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
        }
    }
    PROCTYPE prefix = 0;
    barrier();
    if (my_tile != 0) {
        prefix = shared_prefix;
    }

    // step 6: perform partition-wide scan
    if (local_ix >> 5 > 0) {
        prefix += chunks[(local_ix >> 5) - 1];
    }
    for (uint i = 0; i < ELEMENTS_PER_WORKGROUP; i++) {
        uint ix = (local_ix & 0x3e0) * ELEMENTS_PER_WORKGROUP + i * 32 + (local_ix & 0x1f);
        PROCTYPE agg = aggregates[i];
        //outbuf[mem_base + ix] = prefix + agg;
        outbuf[mem_base + ix] = 1;
        prefix += subgroupBroadcast(agg, 31);
    }
}
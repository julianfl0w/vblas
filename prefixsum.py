import os
import sys

localtest = True
here = os.path.dirname(os.path.abspath(__file__))
if localtest == True:
    sys.path = [os.path.join(here, "..", "vulkanese", "vulkanese")] + sys.path

from vulkanese import *
import numpy as np
from platform_constants import default_constants as platformConstantsDict

here = os.path.dirname(os.path.abspath(__file__))


class PREFIX_SUM(ComputeShader):
    def __init__(
        self,
        constantsDict,
        instance,
        device,
        X,
        devnum=0,
        DEBUG=False,
        buffType="float64_t",
        memProperties=0
        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    ):

        constantsDict["PROCTYPE"] = buffType
        constantsDict["XDIM0"] = np.shape(X)[0]
        constantsDict["LG_WG_SIZE"] = 9 # corresponding to 512 threads per NVIDIA SIMD
        constantsDict["SHADERS_PER_LOCALGROUP"] = (1 << constantsDict["LG_WG_SIZE"])
        constantsDict["SHADER_COUNT"] = 512
        constantsDict["OPS_PER_SHADER"] = 1
        
        self.dim2index = {
            "XDIM0": np.shape(X)[0],
            "SHADER_COUNT": constantsDict["SHADER_COUNT"],
        }

        # device selection and instantiation
        self.instance_inst = instance
        self.device = device
        self.constantsDict = constantsDict
        shader_basename = "shaders/prefix"


        shaderInputBuffers = [
        ]
        shaderInputBuffersNoDebug = []
        debuggableVars = [
            {"name": "thisAdd", "type": buffType, "dims": ["XDIM0", "XDIM1"]}
        ]
        shaderOutputBuffers = [
            {"name": "inbuf", "type": buffType, "dims": ["XDIM0"]},
            {"name": "outbuf", "type": buffType, "dims": ["XDIM0"]},
            {"name": "part_counter", "type": "uint", "dims": ["SHADER_COUNT"]},
            {"name": "localGroup_flag", "type": "uint", "dims": ["SHADER_COUNT"]},
            {"name": "aggregate", "type": buffType, "dims": ["SHADER_COUNT"]},
            {"name": "prefix", "type": buffType, "dims": ["SHADER_COUNT"]},
        ]

        # Compute Stage: the only stage
        ComputeShader.__init__(
            self,
            sourceFilename=os.path.join(here, "shaders/prefix.c"),  # can be GLSL or SPIRV
            parent=self.instance_inst,
            constantsDict=self.constantsDict,
            device=self.device,
            name=shader_basename,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            shaderInputBuffers=shaderInputBuffers,
            shaderInputBuffersNoDebug=shaderInputBuffersNoDebug,
            debuggableVars=debuggableVars,
            shaderOutputBuffers=shaderOutputBuffers,
            DEBUG=DEBUG,
            dim2index=self.dim2index,
            memProperties=memProperties,
            workgroupShape=[1, 1, 1],
            compressBuffers=True,
        )

    def debugRun(self):
        self.aggregate.zeroInitialize()
        self.prefix.zeroInitialize()
        self.flag.zeroInitialize()
        self.part_counter.zeroInitialize()
        vstart = time.time()
        self.run()
        vlen = time.time() - vstart
        print("vlen " + str(vlen))
        return self.Z.getAsNumpyArray()


import time


def numpyTest(X):

    # get numpy time, for comparison
    print("--- RUNNING NUMPY TEST ---")
    for i in range(10):
        nstart = time.time()
        nval = np.sum(X)
        nlen = time.time() - nstart
        print("nlen " + str(nlen))
    return nval


def floatTest(X, instance, expectation):

    print("--- RUNNING FLOAT TEST ---")
    devnum = 0
    device = instance.getDevice(devnum)
    s = PREFIX_SUM(
        platformConstantsDict,
        instance=instance,
        device=device,
        X=X.astype(np.float32),
        buffType="float",
        memProperties=0 | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        # | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )
    s.inbuf.setBuffer(X)
    for i in range(10):
        vval = s.debugRun()
    print(np.allclose(expectation, vval))
    device.release()


def float64Test(X, instance, expectation):

    print("--- RUNNING FLOAT64_T TEST ---")
    devnum = 0
    device = instance.getDevice(devnum)
    s = PREFIX_SUM(
        platformConstantsDict,
        instance=instance,
        device=device,
        X=X,
        buffType="float64_t",
        memProperties=0
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )
    s.inbuf.setBuffer(X)
    for i in range(10):
        vval = s.debugRun()
    print(np.allclose(expectation, vval))
    device.release()


if __name__ == "__main__":

    signalLen = 2 ** 13
    wcount = 512
    X = np.random.random((wcount, signalLen))
    Y = np.random.random((signalLen))

    # begin GPU test
    instance = Instance(verbose=False)
    nval = numpyTest(X)
    floatTest(X, instance, expectation=nval)
    float64Test(X, instance, expectation=nval)

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


class ADD(ComputeShader):
    def __init__(
        self,
        constantsDict,
        instance,
        device,
        X,
        Y,
        devnum=0,
        DEBUG=False,
        buffType="float64_t",
        memProperties=0
        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    ):

        constantsDict["PROCTYPE"] = buffType
        constantsDict["YLEN"] = np.prod(np.shape(Y))
        constantsDict["LG_WG_SIZE"] = 6 # corresponding to 512 threads per NVIDIA SIMD
        constantsDict["THREADS_PER_LOCALGROUP"] = (1 << constantsDict["LG_WG_SIZE"])
        constantsDict["OPS_PER_THREAD"] = 1
        self.dim2index = {
        }

        # device selection and instantiation
        self.instance = instance
        self.device = device
        self.constantsDict = constantsDict
        shader_basename = "shaders/add"


        shaderInputBuffers = [
        ]
        shaderInputBuffersNoDebug = []
        debuggableVars = [
        ]
        shaderOutputBuffers = [
            {"name": "x", "type": buffType, "dims": np.shape(X), "qualifier": "readonly"},
            {"name": "y", "type": buffType, "dims": np.shape(Y), "qualifier": "readonly"},
            {"name": "sumOut", "type": buffType, "dims": np.shape(X), "qualifier": "writeonly"},
        ]

        # Compute Stage: the only stage
        ComputeShader.__init__(
            self,
            sourceFilename=os.path.join(here, "shaders/add.c"),  # can be GLSL or SPIRV
            parent=self.instance,
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
            workgroupShape=[int(np.prod(np.shape(X))/(constantsDict["THREADS_PER_LOCALGROUP"]*constantsDict["OPS_PER_THREAD"])), 1, 1],
            compressBuffers=True,
        )

    def debugRun(self):
        vstart = time.time()
        self.run()
        vlen = time.time() - vstart
        print("vlen " + str(vlen))
        #return self.sumOut.getAsNumpyArray()


import time


def numpyTest(X, Y):

    # get numpy time, for comparison
    print("--- RUNNING NUMPY TEST ---")
    for i in range(10):
        nstart = time.time()
        nval = np.add(X,Y)
        nlen = time.time() - nstart
        print("nlen " + str(nlen))
    return nval


def floatTest(X, Y, instance, expectation):

    print("--- RUNNING FLOAT TEST ---")
    devnum = 0
    device = instance.getDevice(devnum)
    s = ADD(
        platformConstantsDict,
        instance=instance,
        device=device,
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        buffType="float",
        memProperties=0 | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        # | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )
    s.x.setBuffer(X)
    s.y.setBuffer(Y)
    for i in range(10):
        #vval = s.debugRun()
        s.debugRun()
    vval = s.sumOut.getAsNumpyArray()
    print(np.allclose(expectation, vval))
    device.release()


def float64Test(X, Y, instance, expectation):

    print("--- RUNNING FLOAT64_T TEST ---")
    devnum = 0
    device = instance.getDevice(devnum)
    s = ADD(
        platformConstantsDict,
        instance=instance,
        device=device,
        X=X,
        Y=Y,
        buffType="float64_t",
        memProperties=0
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )
    s.x.setBuffer(X)
    s.y.setBuffer(Y)
    for i in range(10):
        #vval = s.debugRun()
        s.debugRun()
    #print(np.allclose(expectation, vval))
    device.release()


if __name__ == "__main__":

    signalLen = 2 ** 13
    wcount = 512
    signalLen = 2 ** 23
    wcount = 1
    X = np.random.random((wcount, signalLen))
    Y = np.random.random((signalLen))

    # begin GPU test
    instance = Instance(verbose=False)
    nval = numpyTest(X, Y)
    floatTest(X, Y, instance, expectation=nval)
    float64Test(X, Y, instance, expectation=nval)

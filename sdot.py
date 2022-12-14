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


class SDOT(ComputeShader):
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
        constantsDict["XDIM0"] = np.shape(X)[0]
        constantsDict["XDIM1"] = np.shape(X)[1]
        constantsDict["YDIM0"] = np.shape(Y)[0]

        if np.shape(X)[-1] != np.shape(Y)[0]:
            raise Exception("Last dimension of X must match first dimension of Y")

        self.dim2index = {
            "XDIM0": np.shape(X)[0],
            "XDIM1": np.shape(X)[1],
            "YDIM0": np.shape(Y)[0],
        }

        # device selection and instantiation
        self.instance_inst = instance
        self.device = device
        self.constantsDict = constantsDict
        shader_basename = "sdot"
        self.dim2index = {"XDIM0": "w", "XDIM1": "n"}

        shaderInputBuffers = []
        shaderInputBuffersNoDebug = []
        debuggableVars = [
            {"name": "thisAdd", "type": buffType, "dims": ["XDIM0", "XDIM1"]}
        ]
        shaderOutputBuffers = [
            {"name": "Z", "type": buffType, "dims": ["XDIM0"]},
            {"name": "X", "type": buffType, "dims": ["XDIM0", "XDIM1"]},
            {"name": "Y", "type": buffType, "dims": ["YDIM0"]},
        ]

        # Compute Stage: the only stage
        ComputeShader.__init__(
            self,
            sourceFilename=os.path.join(here, "sdot.c"),  # can be GLSL or SPIRV
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
        vstart = time.time()
        self.run()
        vlen = time.time() - vstart
        print("vlen " + str(vlen))
        return self.Z.getAsNumpyArray()


import time


def numpyTest(X, Y):

    # get numpy time, for comparison
    print("--- RUNNING NUMPY TEST ---")
    for i in range(10):
        nstart = time.time()
        nval = np.dot(X, Y)
        nlen = time.time() - nstart
        print("nlen " + str(nlen))
    return nval


def floatTest(X, Y, instance, expectation):

    print("--- RUNNING FLOAT TEST ---")
    devnum = 0
    device = instance.getDevice(devnum)
    s = SDOT(
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
    s.X.setBuffer(X)
    s.Y.setBuffer(Y)
    for i in range(10):
        vval = s.debugRun()
    print(np.allclose(expectation, vval))
    device.release()


def float64Test(X, Y, instance, expectation):

    print("--- RUNNING FLOAT64_T TEST ---")
    devnum = 0
    device = instance.getDevice(devnum)
    s = SDOT(
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
    s.X.setBuffer(X)
    s.Y.setBuffer(Y)
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
    nval = numpyTest(X, Y)
    floatTest(X, Y, instance, expectation=nval)
    float64Test(X, Y, instance, expectation=nval)

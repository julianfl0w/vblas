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

        constantsDict["CAST_PRECISION"] = 100000
        constantsDict["PROCTYPE"] = buffType
        constantsDict["LOG2_THREADS_PER_WORKGROUP"] = 7
        constantsDict["THREADS_PER_WORKGROUP"] = (
            1 << constantsDict["LOG2_THREADS_PER_WORKGROUP"]
        )
        constantsDict["OPS_PER_THREAD"] = 1
        constantsDict["WORKGROUP_COUNT"] = int(
            np.prod(np.shape(X))
            / (constantsDict["THREADS_PER_WORKGROUP"] * constantsDict["OPS_PER_THREAD"])
        )
        print(constantsDict["WORKGROUP_COUNT"])
        constantsDict["THREADS_PER_DISPATCH"] = (
            constantsDict["THREADS_PER_WORKGROUP"] * constantsDict["WORKGROUP_COUNT"]
        )

        # device selection and instantiation
        self.instance_inst = instance
        self.device = device
        self.constantsDict = constantsDict
        shader_basename = "shaders/prefix"
        self.dim2index = {}

        memProperties = (
            0
            | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )

        buffers = [
            StorageBuffer(
                device=self.device,
                name="inbuf",
                memtype=buffType,
                qualifier="readonly",
                dimensionNames=np.shape(X),
                dimensionVals=np.shape(X),
                memProperties=memProperties,
            ),
            StorageBuffer(
                device=self.device,
                name="outbuf",
                memtype=buffType,
                qualifier="writeonly",
                dimensionNames=np.shape(X),
                dimensionVals=np.shape(X),
                memProperties=memProperties,
            ),
            
            # work_buf[0] is the tile id
            # work_buf[i * 4 + 1] is the flag for tile i
            # work_buf[i * 4 + 2] is the aggregate for tile i
            # work_buf[i * 4 + 3] is the prefix for tile i

            StorageBuffer(
                device=self.device,
                name="work_buf",
                memtype="uint",
                qualifier="",
                dimensionNames=["??"],
                dimensionVals=[constantsDict["WORKGROUP_COUNT"]*4],
                memProperties=memProperties,
            ),
        ]

        if DEBUG:
            buffers += [
                DebugBuffer(
                    device=self.device,
                    name="thisAdd",
                    dimensionNames=np.shape(X),
                    dimensionVals=np.shape(X),
                    memProperties=memProperties,
                )
            ]

        # Compute Stage: the only stage
        ComputeShader.__init__(
            self,
            sourceFilename=os.path.join(
                here, "shaders/prefix.c"
            ),  # can be GLSL or SPIRV
            parent=self.instance_inst,
            constantsDict=self.constantsDict,
            device=self.device,
            name=shader_basename,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            buffers=buffers,
            DEBUG=DEBUG,
            dim2index=self.dim2index,
            memProperties=memProperties,
            workgroupShape=[constantsDict["WORKGROUP_COUNT"], 1, 1],
            compressBuffers=True,
        )

    def debugRun(self):
        vstart = time.time()
        self.run()
        vlen = time.time() - vstart
        print("vlen " + str(vlen))
        return self.outbuf.getAsNumpyArray()


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
        
    print(expectation)
    print(vval)
    print(np.sum(vval))
    print(np.allclose(expectation, vval/s.constantsDict["CAST_PRECISION"]))
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
    )
    s.inbuf.setBuffer(X)
    for i in range(10):
        vval = s.debugRun()
    print(vval)
    print(np.allclose(expectation, vval/s.constantsDict["CAST_PRECISION"]))
    device.release()


if __name__ == "__main__":

    signalLen = 1024
    X = np.arange(signalLen)

    # begin GPU test
    instance = Instance(verbose=False)
    nval = numpyTest(X)
    floatTest(X, instance, expectation=nval)
    #float64Test(X, instance, expectation=nval)

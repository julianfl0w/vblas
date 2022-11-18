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


class ARITH(ComputeShader):
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
        shader_basename = "shaders/arith",
        memProperties=0
        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    ):

        constantsDict["PROCTYPE"] = buffType
        constantsDict["YLEN"] = np.prod(np.shape(Y))
        constantsDict["LG_WG_SIZE"] = 7 # corresponding to 512 threads per NVIDIA SIMD
        constantsDict["THREADS_PER_WORKGROUP"] = (1 << constantsDict["LG_WG_SIZE"])
        constantsDict["OPS_PER_THREAD"] = 1
        self.dim2index = {
        }

        # device selection and instantiation
        self.instance = instance
        self.device = device
        self.constantsDict = constantsDict


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
            sourceFilename=os.path.join(here, shader_basename + ".c"),  # can be GLSL or SPIRV
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
            workgroupShape=[int(np.prod(np.shape(X))/(constantsDict["THREADS_PER_WORKGROUP"]*constantsDict["OPS_PER_THREAD"])), 1, 1],
            compressBuffers=True,
        )

    def debugRun(self):
        vstart = time.time()
        self.run()
        vlen = time.time() - vstart
        print("vlen " + str(vlen))
        #return self.sumOut.getAsNumpyArray()

class ADD(ARITH):
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
        constantsDict["OPERATION"] = "+"
        ARITH.__init__(
          self, 
          constantsDict   =  constantsDict,
          instance        =  instance,
          device          =  device,
          X               =  X,
          Y               =  Y,
          devnum          =  devnum,
          DEBUG           =  DEBUG,
          buffType        =  buffType,
          shader_basename =  "shaders/arith",
          memProperties   =  memProperties)

class MULTIPLY(ARITH):
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
        constantsDict["OPERATION"] = "*"
        ARITH.__init__(
          self, 
          constantsDict   =  constantsDict,
          instance        =  instance,
          device          =  device,
          X               =  X,
          Y               =  Y,
          devnum          =  devnum,
          DEBUG           =  DEBUG,
          buffType        =  buffType,
          shader_basename =  "shaders/arith",
          memProperties   =  memProperties)
        
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


def numpyTestMult(X, Y):

    # get numpy time, for comparison
    print("--- RUNNING Mult NUMPY TEST ---")
    for i in range(10):
        nstart = time.time()
        nval = np.multiply(X,Y)
        nlen = time.time() - nstart
        print("nlen " + str(nlen))
    return nval


def floatTestMult(X, Y, instance, expectation):

    print("--- RUNNING Mult FLOAT TEST ---")
    devnum = 0
    device = instance.getDevice(devnum)
    s = MULTIPLY(
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
    #print("expectation")
    #print(json.dumps(expectation.flatten()[:256].tolist()))
    #print("vval")
    #print(json.dumps(vval.flatten()[:256].tolist()))
    #print(np.allclose(expectation, vval))
    device.release()


def float64TestMult(X, Y, instance, expectation):

    print("--- RUNNING Mult FLOAT64_T TEST ---")
    devnum = 0
    device = instance.getDevice(devnum)
    s = MULTIPLY(
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

    wcount = 512
    signalLen = 2 ** 23
    
    wcount = 1
    signalLen = 128*64
    signalLen = 2 ** 23
    X = np.random.random((wcount, signalLen))
    Y = np.random.random((signalLen))

    # begin GPU test
    instance = Instance(verbose=False)
    nval = numpyTest(X, Y)
    floatTest(X, Y, instance, expectation=nval)
    float64Test(X, Y, instance, expectation=nval)
    
    # multiply test
    nval = numpyTestMult(X, Y)
    floatTestMult(X, Y, instance, expectation=nval)
    float64TestMult(X, Y, instance, expectation=nval)

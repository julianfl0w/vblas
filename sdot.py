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
class SDOT:
    def __init__(self, constantsDict, X, Y, devnum = 0, DEBUG = False, buffType = "float64_t"):
         
        if np.shape(X)[-1] != np.shape(Y)[0]:
            raise Exception("Last dimension of X must match first dimension of Y")

        self.dim2index = {
            "XDIM0": np.shape(X)[0],
            "XDIM1": np.shape(X)[1],
            "YDIM0": np.shape(Y)[0],
        }
        
        # device selection and instantiation
        self.instance_inst = Instance(verbose=False)
        self.device = self.instance_inst.getDevice(devnum)
        self.constantsDict = constantsDict
        shader_basename = "sdot"
        self.dim2index  = {
            "XDIM0": "w",
            "XDIM1": "n",
        }
        
        shaderInputBuffers=[
            {"name": "X", "type": buffType, "dims": ["XDIM0", "XDIM1"]},
            {"name": "Y", "type": buffType, "dims": ["YDIM0"]},
        ]
        shaderInputBuffersNoDebug=[
        ]
        debuggableVars=[
            {"name": "thisAdd", "type": buffType, "dims": ["XDIM0", "XDIM1"]},
        ]
        shaderOutputBuffers=[
            {"name": "Z", "type": buffType, "dims": ["XDIM0"]},
        ]

        # Compute Stage: the only stage
        self.computeShader = Stage(
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
        )

        # generate a compute cmd buffer
        self.computePipeline = ComputePipeline(
            computeShader=self.computeShader,
            device=self.device,
            constantsDict=self.constantsDict,
            workgroupShape=[128, 1, 1],
        )
    
    def run(self):
        
        self.computePipeline.run()

import time

if __name__ == "__main__":
    
    X = np.random.random((1000,1000))
    Y = np.random.random((1000))
    
    platformConstantsDict["XDIM0"] = np.shape(X)[0]
    platformConstantsDict["XDIM1"] = np.shape(X)[1]
    platformConstantsDict["YDIM0"] = np.shape(Y)[0]
    
    s = SDOT(platformConstantsDict, X = X, Y = Y)
    s.computeShader.X.setBuffer(X.flatten())
    s.computeShader.Y.setBuffer(Y)
    vstart = time.time()
    s.run()
    vlen = time.time() - vstart
    nstart = time.time()
    nval = np.dot(X,Y)
    nlen = time.time() - nstart
    vval = s.computeShader.Z.getAsNumpyArray()
    #s.computeShader.dumpMemory(os.path.join(here, "debug.json"))
    print("vlen " + str(vlen))
    print("nlen " + str(nlen))
    print(np.allclose(nval, vval))
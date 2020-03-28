import os
import sys
import numpy as np
import tvm
from tvm.contrib import graph_runtime

### TODO: add arguments for artifacts folders
imageIN = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dog.npy')).astype(np.float32)
imageIN = np.squeeze(imageIN, axis=0)
print(imageIN.shape)
image = np.concatenate([imageIN[np.newaxis, :, :]]*1)
image = image * 0.25
#image = image.transpose(0, 2, 3, 1)
print(image.shape)

#data_shape=(1, 3, 224, 224)  # how to get this automatically?
#input_data = tvm.nd.array(np.random.uniform(-1,1,size=data_shape).astype("float32"))
#input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))
input_data = image

print("----- Running inference on ARM -----")
loaded_json_arm = open("./artifacts_arm/deploy_graph.json").read()
loaded_lib_arm =tvm.runtime.load_module("./artifacts_arm/deploy_lib.so")
loaded_params_arm = bytearray(open("./artifacts_arm/deploy_param.params", "rb").read())

ctx_arm=tvm.cpu()
module_arm = graph_runtime.create(loaded_json_arm, loaded_lib_arm, ctx_arm)
module_arm.load_params(loaded_params_arm)
module_arm.run(data=input_data)
out_deploy_arm = module_arm.get_output(0).asnumpy()
print(out_deploy_arm.shape)
#print(out_deploy_arm.flatten()[0:100])
#print(out_deploy_arm)
#np.savetxt('out_arm.txt', out_deploy_arm.flatten(), fmt='%10.5f')

print("----- Running inference on TIDL -----")
loaded_json=open("./artifacts_hetero/deploy_graph.json").read()
loaded_lib =tvm.runtime.load_module("./artifacts_hetero/deploy_lib.so")
loaded_params = bytearray(open("./artifacts_hetero/deploy_param.params", "rb").read())

ctx=tvm.cpu()
module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
module.run(data=input_data)
out_deploy = module.get_output(0).asnumpy()
print(out_deploy.shape)
#print(out_deploy.flatten()[0:100])
#print(out_deploy)
np.savetxt('out_tidl.txt', out_deploy.flatten(), fmt='%10.5f')

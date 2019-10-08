# Usage
# nvprof --profile-from-start-off python3 test_tensorrt.py 2>&1 | tee profile.txt
import numpy as np
import time
from mxnet.gluon.model_zoo.vision import get_model

import tvm
from tvm import relay
import tvm.relay.testing
import tvm.relay.transform

def test_extern_tensorrt():
    dtype = 'float32'
    xshape = (1, 32, 14, 14)
    yshape = (1, 32,  1,  1)
    zshape = (1,  1,  1,  1)
    x = relay.var('x', shape=(xshape), dtype=dtype)
    y = relay.var('y', shape=(yshape), dtype=dtype)
    z = relay.var('z', shape=(zshape), dtype=dtype)
    w = z * (x + y)
    out = relay.nn.relu(w)
    f = relay.Function([x, y, z], out)

    mod = relay.Module()
    mod['main'] = WholeGraphAnnotator('tensorrt').visit(f)
    mod = relay.transform.PartitionGraph()(mod)

    ref_mod = relay.Module()
    ref_mod['main'] = f

    x_data = np.random.uniform(-1, 1, xshape).astype(dtype)
    y_data = np.random.uniform(-1, 1, yshape).astype(dtype)
    z_data = np.random.uniform(-1, 1, zshape).astype(dtype)

    # Test against reference.
    for kind in ["vm"]: # ["vm" , "debug"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.gpu(0), target='cuda')
        # First execution will trigger build of TRT engine(s).
        res = ex.evaluate()(x_data, y_data, z_data)
        # TRT engine is reused for second execution.
        res = ex.evaluate()(x_data, y_data, z_data)

        ref_ex = relay.create_executor(kind, mod=ref_mod, ctx=tvm.cpu(0))
        ref_res = ref_ex.evaluate()(x_data, y_data, z_data)

        tvm.testing.assert_allclose(res.asnumpy(), ref_res.asnumpy(), rtol=1e-5)

    print('Test passed.')

def test_extern_tensorrt_mobilenet():
    # FIXME: This test is only for demo purpose and supposed to be removed.
    dtype = 'float32'
    input_shape = (1, 3, 224, 224)
    #mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype='float32')
    from mxnet.gluon.model_zoo.vision import get_model
    block = get_model('resnet50_v1', pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)

    # mod = relay.transform.ExternOp('tensorrt')(mod)
    # mod = relay.transform.PartitionGraph()(mod)

    mod['main'] = WholeGraphAnnotator('tensorrt').visit(mod['main'])
    mod = relay.transform.PartitionGraph()(mod)

    i_data = np.random.uniform(0, 1, input_shape).astype(dtype)

    for kind in ["vm"]: #["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.gpu(0), target='cuda')
        res = ex.evaluate()(i_data, **params)

        times = []
        for i in range(10):
            start_time = time.time()
            res = ex.evaluate()(i_data, **params)
            times.append(time.time() - start_time)
        print('Mean latency', np.mean(times)*1000)

    # FIXME: When subgraph has only one op, Relay executor will use the cache value instead
    # of re-computing, so the following checking logic does not work.
    ref_mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype='float32')
    ref_ex = relay.create_executor("vm", mod=ref_mod, ctx=tvm.cpu(0))
    ref_res = ref_ex.evaluate()(i_data, **params)

    tvm.testing.assert_allclose(res.asnumpy(), ref_res.asnumpy(), rtol=1e-5)


def test_extern_tensorrt_perf(model='resnet50_v1', use_trt=True, profile=True, num_iteration=1000):
    if profile:
        import ctypes
        _cudart = ctypes.CDLL('libcudart.so')

    dtype = 'float32'
    input_shape = (1, 3, 224, 224)
    block = get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)

    if use_trt:
        from tvm.relay.annotation import subgraph_begin, subgraph_end
        from test_pass_partition_graph import WholeGraphAnnotator
        mod['main'] = WholeGraphAnnotator('tensorrt').visit(mod['main'])
        mod = relay.transform.PartitionGraph()(mod)

    i_data = np.random.uniform(0, 1, input_shape).astype(dtype)

    if use_trt:
        with relay.build_config(opt_level=2):
            vm = tvm.relay.vm.compile(mod, 'cuda')
            vm.init(tvm.gpu(0))
            vm.load_params(params)
    else:
        with relay.build_config(opt_level=3):
            vm = tvm.relay.vm.compile(mod, 'cuda', params=params)
            vm.init(tvm.gpu(0))
            vm.load_params(params)
    
    # Warmup
    for i in range(10):
        vm.invoke('main', i_data)

    # Start profiling
    if profile:
        ret = _cudart.cudaProfilerStart()
        if ret != 0:
            raise Exception("cudaProfilerStart() returned %d" % ret)

    # Time
    start_time = time.time()
    for i in range(num_iteration):
        vm.invoke('main', i_data)
    end_time = time.time()
    latency = (end_time-start_time)/num_iteration*1000
    print(model, use_trt, latency)
    return latency

if __name__ == "__main__":
    # test_extern_tensorrt()
    latency = {}
    models = [
        'alexnet',
        'resnet18_v1',
        'resnet34_v1',
        'resnet50_v1',
        'resnet101_v1',
        'resnet152_v1',
        'resnet18_v2',
        'resnet34_v2',
        'resnet50_v2',
        'resnet101_v2',
        'resnet152_v2',
        'squeezenet1.0',
        'mobilenet0.25',
        'mobilenet0.5',
        'mobilenet0.75',
        'mobilenet1.0',
        'mobilenetv2_0.25',
        'mobilenetv2_0.5',
        'mobilenetv2_0.75',
        'mobilenetv2_1.0',
        'vgg11',
        'vgg16',
        'densenet121',
        'densenet169',
        'densenet201'
        ]
    for model in models:
        latency[model] = test_extern_tensorrt_perf(model=model)
    
    for model in models:
        print(model, latency[model])
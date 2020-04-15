import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.expr import Call, Constant, Tuple, GlobalVar
from tvm.relay.op import Op
from tvm.relay.function import Function

from tvm.contrib import graph_runtime
from tvm.relay.op.annotation import compiler_begin, compiler_end
import numpy as np

class VarReplacer(ExprMutator):
    def __init__(self, var_map):
        ExprMutator.__init__(self)
        self.var_map = var_map

    def visit_var(self, var):
        if var in self.var_map:
            return self.var_map[var]
        return super().visit_var(var)

class SubgraphRemover(ExprMutator):
    def __init__(self, subgraphs_to_remove, mod, new_mod):
        ExprVisitor.__init__(self)
        self.subgraphs_to_remove = subgraphs_to_remove
        self.mod = mod
        self.new_mod = new_mod

    def visit_call(self, call):
        if isinstance(call.op, GlobalVar):
            name = call.op.name_hint
            if name in self.subgraphs_to_remove:
                # "Inline" the subgraph back into new main function.
                func = self.mod[name]
                var_map = {}
                for arg, param in zip(call.args, func.params):
                    var_map[param] = super().visit(arg)
                new_body = VarReplacer(var_map).visit(func.body)
                return new_body
            elif name != "main":
                # Copy the GlobalVar (subgraph function) to the new module and call.
                args = []
                for arg in call.args:
                    args.append(super().visit(arg))
                subgraph_gv = relay.GlobalVar(name)
                self.new_mod[subgraph_gv] = self.mod[name]
                return subgraph_gv(*args)
        return super().visit_call(call)

def PruneSubgraphs(mod, compiler="tidl", num_subgraphs_to_keep=4):
    subgraph_with_macs = []
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        num_macs = relay.analysis.get_total_mac_number(mod[name])
        subgraph_with_macs.append([name, num_macs])
    print("Subgraphs with computed # of MACS:", subgraph_with_macs)
    subgraphs_to_remove = sorted(subgraph_with_macs, key=lambda x: int(x[1]))[:-num_subgraphs_to_keep]
    print("Will remove these subgraphs:", subgraphs_to_remove)
    subgraph_names_to_remove = set([x[0] for x in subgraphs_to_remove])
    # Create new pruned module
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod).visit(mod["main"])
    return new_mod

"""
def test_remove_subgraphs():
    dtype = 'float32'
    xshape = (1, 8, 5, 5)
    w0_shape = (4, 8,  1,  1)
    w1_shape = (4, 4,  1,  1)
    w2_shape = (2, 4,  1,  1)
    w3_shape = (2, 2,  1,  1)
    x = relay.var('x', shape=(xshape), dtype=dtype)
    w0 = relay.var('w0', shape=(w0_shape), dtype=dtype)
    a = relay.nn.conv2d(x, w0, kernel_size=1)
    w1 = relay.var('w1', shape=(w1_shape), dtype=dtype)
    b = relay.nn.conv2d(a, w1, kernel_size=1)
    w2 = relay.var('w2', shape=(w2_shape), dtype=dtype)
    c = relay.nn.conv2d(b, w2, kernel_size=1)
    w3 = relay.var('w3', shape=(w3_shape), dtype=dtype)
    out = relay.nn.conv2d(c, w3, kernel_size=1)
    f = relay.Function([x, w0, w1, w2, w3], out)

    mod = tvm.IRModule()
    mod['main'] = f

    mod = WhiteListAnnotator(['nn.conv2d'], "tidl")(mod)
    mod = relay.transform.PartitionGraph()(mod)
    mod = relay.transform.InferType()(mod)
    print("===================")
    print("Partitioned module:")
    print(mod)
    print("===================")

    mod = PruneSubgraphs(mod, compiler="tidl", num_subgraphs_to_keep=2)

    print("===================")
    print("Partitioned module after pruning:")
    print(mod)
    print("===================")

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, "llvm")
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    x_data = np.random.uniform(-1, 1, xshape).astype(dtype)
    w0_data = np.random.uniform(-1, 1, w0_shape).astype(dtype)
    w1_data = np.random.uniform(-1, 1, w1_shape).astype(dtype)
    w2_data = np.random.uniform(-1, 1, w2_shape).astype(dtype)
    w3_data = np.random.uniform(-1, 1, w3_shape).astype(dtype)
    mod.run(x=x_data, w0=w0_data, w1=w1_data, w2=w2_data, w3=w3_data)
    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]

test_remove_subgraphs()
"""
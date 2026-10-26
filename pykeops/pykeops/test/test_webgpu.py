import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")
if wgpu.gpu.request_adapter_sync() is None:
    pytest.skip("No WebGPU adapter available", allow_module_level=True)

from pykeops.numpy import Genred, LazyTensor

rng = np.random.default_rng(0)


@pytest.fixture(autouse=True)
def seed():
    global rng
    rng = np.random.default_rng(0)


def rand(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def compare_to_cpu(formula, aliases, args, reduction_op="Sum", axis=1, tol=1e-4):
    routine = Genred(formula, aliases, reduction_op=reduction_op, axis=axis)
    ref = routine(*args, backend="CPU")
    out = routine(*args, backend="WebGPU")
    # N.B. Min_ArgMin and Max_ArgMax reductions return a tuple of arrays
    if not isinstance(ref, tuple):
        out, ref = (out,), (ref,)
    for out_k, ref_k in zip(out, ref):
        assert out_k.shape == ref_k.shape
        np.testing.assert_allclose(out_k, ref_k, rtol=tol, atol=tol)


# M, N : sizes not multiple of the block size, and N smaller than the block size
@pytest.mark.parametrize("M, N", [(1003, 2511), (1, 1), (517, 7)])
@pytest.mark.parametrize("axis", [0, 1])
def test_gaussian_conv(M, N, axis):
    aliases = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)", "p=Pm(1)"]
    if axis == 0:
        aliases[2] = "b=Vi(2)"
    args = (
        rand(M, 3),
        rand(N, 3),
        rand(M if axis == 0 else N, 2),
        np.array([0.5], dtype=np.float32),
    )
    compare_to_cpu("Exp(-p*SqDist(x,y))*b", aliases, args, axis=axis)


@pytest.mark.parametrize(
    "reduction_op",
    ["Sum", "Max", "Min", "ArgMax", "ArgMin", "Min_ArgMin", "Max_ArgMax", "LogSumExp"],
)
def test_reductions(reduction_op):
    formula = "SqDist(x,y)" if "Arg" in reduction_op else "-SqDist(x,y)"
    aliases = ["x=Vi(3)", "y=Vj(3)"]
    compare_to_cpu(
        formula, aliases, (rand(300, 3), rand(700, 3)), reduction_op=reduction_op
    )


def test_logsumexp_weighted():
    aliases = ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)"]
    args = (rand(300, 3), rand(700, 3), np.abs(rand(700, 2)))
    compare_to_cpu("-SqDist(x,y)", aliases, args, reduction_op="LogSumExp", axis=1)
    routine = Genred(
        "-SqDist(x,y)", aliases, reduction_op="LogSumExp", axis=1, formula2="b"
    )
    np.testing.assert_allclose(
        routine(*args, backend="WebGPU"),
        routine(*args, backend="CPU"),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "formula",
    [
        "Exp(x)+Log(Abs(y))+Sqrt(Abs(x))+Rsqrt(Abs(y)+1)+Inv(y+10)",
        "Sin(x)+Cos(y)+Acos(x/10)+Asin(y/10)+Atan(x)+Atan2(x,y)",
        "Pow(x,3)+Pow(y,2)+Powf(Abs(x),y)+Square(x)+IntInv(3)*x",
        "ReLU(x)+Step(y)+Sign(x)+Abs(y)+Floor(x)",
        "Clamp(x,y,y+1)+ClampInt(x,-1,1)+DiffClampInt(x,-1,1)+IfElse(x,y,x)",
        "Mod(x,y+5,x)+Round(x,2)+Round(y,0)",
        "Equal(Floor(x),Floor(y))+NotEqual(x,y)+LessThan(x,y)+LessOrEqual(x,y)",
        "Concat(Extract(x,1,2),Elem(y,0))",
        "SumT(Sum(x),3)*ElemT(Elem(y,1),3,1)",
        "Norm2(x)*(x|y)*Normalize(y)",
        "Max(x)*Min(y)*ArgMax(x)*ArgMin(y)",
        "Grad(Exp(-SqDist(x,y)),x,e)",
        "IntCst(3)*x+RatCst(1,3)*y-x/(Abs(y)+1)",
    ],
)
def test_operations(formula):
    aliases = ["x=Vi(3)", "y=Vj(3)", "e=Vi(1)"]
    compare_to_cpu(formula, aliases, (rand(200, 3), rand(300, 3), rand(200, 1)))


def test_sinxdivx():
    # N.B. WebGPU only guarantees an absolute accuracy for sin, which is amplified
    # by the division by x when x is close to 0, hence the larger tolerance.
    aliases = ["x=Vi(3)", "y=Vj(3)", "e=Vi(1)"]
    args = (rand(200, 3), rand(300, 3), rand(200, 1))
    compare_to_cpu("SinXDivX(x)+XLogX(Abs(y))", aliases, args, tol=1e-3)


@pytest.mark.parametrize("reduction_op", ["Sum", "LogSumExp", "ArgMin"])
def test_chunked_dispatch(monkeypatch, reduction_op):
    # long computations are split into several dispatches, over i and j,
    # which we force here with a small number of interactions per dispatch
    from pykeops.common.keops_io.webgpu import LoadKeOps_webgpu

    monkeypatch.setattr(LoadKeOps_webgpu, "max_interactions", 5000)
    aliases = ["x=Vi(3)", "y=Vj(3)"]
    args = (rand(3001, 3), rand(1000, 3))
    compare_to_cpu("-SqDist(x,y)", aliases, args, reduction_op=reduction_op)


def test_large_dimy():
    # large dimension of j variables, forcing a smaller block size
    aliases = ["x=Vi(70)", "y=Vj(70)", "b=Vj(3)"]
    args = (rand(100, 70) / 5, rand(300, 70) / 5, rand(300, 3))
    compare_to_cpu("Exp(-SqDist(x,y))*b", aliases, args)


def test_many_args():
    # more than the 8 storage buffers guaranteed by default in WebGPU
    n = 10
    formula = "+".join(f"x{k}*y{k}" for k in range(n))
    aliases = [f"x{k}=Vi(1)" for k in range(n)] + [f"y{k}=Vj(1)" for k in range(n)]
    args = [rand(50, 1) for _ in range(n)] + [rand(80, 1) for _ in range(n)]
    compare_to_cpu(formula, aliases, args)


def test_zero_formula():
    aliases = ["x=Vi(3)", "y=Vj(3)"]
    compare_to_cpu("Zero(3)", aliases, (rand(10, 3), rand(20, 3)))


def test_lazytensor():
    x, y, b = rand(500, 3), rand(800, 3), rand(800, 2)
    K = (-LazyTensor(x[:, None, :]).sqdist(LazyTensor(y[None, :, :]))).exp()
    np.testing.assert_allclose(
        K.__matmul__(b, backend="WebGPU"),
        K.__matmul__(b, backend="CPU"),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        K.sum(axis=0, backend="WebGPU"),
        K.sum(axis=0, backend="CPU"),
        rtol=1e-4,
        atol=1e-4,
    )


def test_torch_grad():
    torch = pytest.importorskip("torch")
    from pykeops.torch import Genred as TorchGenred

    torch.manual_seed(0)
    x = torch.randn(300, 3, requires_grad=True)
    y, b = torch.randn(400, 3), torch.randn(400, 2)
    routine = TorchGenred(
        "Exp(-SqDist(x,y))*b",
        ["x=Vi(3)", "y=Vj(3)", "b=Vj(2)"],
        reduction_op="Sum",
        axis=1,
    )
    res = {}
    for backend in ("CPU", "WebGPU"):
        out = routine(x, y, b, backend=backend)
        (grad,) = torch.autograd.grad((out**2).sum(), [x])
        res[backend] = (out, grad)
    for out_webgpu, out_cpu in zip(res["WebGPU"], res["CPU"]):
        torch.testing.assert_close(out_webgpu, out_cpu, rtol=1e-4, atol=1e-4)


def test_unsupported():
    routine = Genred("SqDist(x,y)", ["x=Vi(3)", "y=Vj(3)"], reduction_op="Sum", axis=1)
    x, y = rand(10, 3), rand(20, 3)
    with pytest.raises(ValueError, match="float32"):
        routine(x.astype(np.float64), y.astype(np.float64), backend="WebGPU")
    ranges = (
        np.array([[0, 10]], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64),
        np.array([[0, 20]], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="ranges"):
        routine(x, y, backend="WebGPU", ranges=ranges)

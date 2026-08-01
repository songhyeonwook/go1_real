"""내보낸 정책(ONNX 우선, TorchScript 대안) 로더.

scripts/rsl_rl/export_policy_onnx.py 가 만든 policy.onnx / policy.pt 를
받습니다. 입력 (1, 52) float32 → 출력 (1, 12) raw action.
"""

import numpy as np

import config as C


class Policy:
    def __init__(self, path: str):
        self._path = path
        if path.endswith(".onnx"):
            import onnxruntime as ort

            self._sess = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._sess.get_inputs()[0].name
            self._backend = "onnx"
        else:
            import torch

            self._torch = torch
            self._module = torch.jit.load(path, map_location="cpu")
            self._module.eval()
            self._backend = "torchscript"

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.reshape(1, -1).astype(np.float32)
        if x.shape[1] != C.OBS_DIM:
            raise ValueError(
                f"obs dim {x.shape[1]} != {C.OBS_DIM}; "
                "export 시 GO1_ABS_JOINT_OBS=1 이었는지 확인하세요"
            )
        if self._backend == "onnx":
            (out,) = self._sess.run(None, {self._input_name: x})
        else:
            with self._torch.no_grad():
                out = self._module(self._torch.from_numpy(x)).numpy()
        action = np.asarray(out, dtype=np.float64).reshape(-1)
        if action.shape[0] != C.NUM_ACTIONS:
            raise ValueError(f"action dim {action.shape[0]} != 12")
        return action

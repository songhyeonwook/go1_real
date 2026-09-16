"""내보낸 정책 로더 (NumPy .npz / ONNX / TorchScript).

scripts/export_p3_student.py 가 만든 번들을 받습니다. 셋 다 같은 raw 49차원
관측을 소비합니다 — 학습의 obs 스케일(joint_vel x 0.23)은 .pt/.onnx 는 그래프
안에, .npz 는 lstm_weight_ih 에 접혀 들어가 있습니다.

    obs(49) -> LSTM(256) -> MLP [512, 256, 128] elu -> action(12)

LSTM 은 hidden/cell 을 스텝 간에 이어가므로, 에피소드 시작(기립 후 정책 인계)
마다 reset() 을 호출해야 합니다. 학습에서 에피소드가 hidden=0 에서 시작합니다.

Go1 온보드 NX 에는 onnxruntime 도 torch 도 없으므로(Python 3.6.9, 오프라인)
NX 에서는 .npz 백엔드를 씁니다. 개발 PC 에서는 셋 다 동작합니다.
"""

import json
import os

import numpy as np

import config as C


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class AuxHeads:
    """정책 latent(LSTM 출력)에서 부목 길이와 몸통 선속도를 읽는 보조 헤드.

    학습(phase3_student.py)에서 action head 와 같은 latent 를 공유하며, 각각
    GT 부목 길이 / GT base_lin_vel 로 지도학습됐습니다. 관측에는 들어가지
    않고 진단·텔레메트리 용도입니다 — 실기 Go1 는 둘 다 측정할 수 없습니다.
    """

    def __init__(self, path):
        d = np.load(path)
        self._sw, self._sb = d["splint_weight"], d["splint_bias"]
        self._vw, self._vb = d["vel_weight"], d["vel_bias"]
        self._smean, self._sstd = float(d["splint_mean"]), float(d["splint_std"])
        self._vmean, self._vstd = d["vel_mean"], float(d["vel_std"])

    def __call__(self, hidden):
        """(부목 길이 L [m], base_lin_vel [m/s]) — 역정규화된 물리 단위."""
        L = float(self._sw @ hidden + self._sb) * self._sstd + self._smean
        v = (self._vw @ hidden + self._vb) * self._vstd + self._vmean
        return L, v


class Policy:
    def __init__(self, path):
        self._path = path
        self._dir = os.path.dirname(os.path.abspath(path))
        self._lstm = None
        if path.endswith(".npz"):
            self._load_numpy(path)
        elif path.endswith(".onnx"):
            self._load_onnx(path)
        else:
            self._load_torchscript(path)

        if self._in_dim != C.OBS_DIM:
            raise ValueError(
                "정책 입력 {}차원 != 배포 관측 {}차원. config.py / "
                "policy_io.json / 학습 obs 레이아웃을 대조하세요.".format(
                    self._in_dim, C.OBS_DIM))

        aux_path = os.path.join(self._dir, "aux_heads.npz")
        self.aux = AuxHeads(aux_path) if os.path.exists(aux_path) else None
        self.reset()
        self._verify_reference_io()

    # ---- 백엔드 로드 ------------------------------------------------------

    def _load_numpy(self, path):
        data = np.load(path)
        n = int(data["num_mlp_layers"]) if "num_mlp_layers" in data else 4
        self._layers = [
            (data["{}_weight".format(2 * i)].astype(np.float32),
             data["{}_bias".format(2 * i)].astype(np.float32))
            for i in range(n)
        ]
        if "lstm_weight_ih" not in data:
            raise ValueError("{}: LSTM 가중치가 없습니다 — phase3 student 번들이 "
                             "아닙니다.".format(path))
        gate_order = (data["gate_order"].tobytes().decode()
                      if "gate_order" in data else "ifgo")
        if gate_order != "ifgo":
            raise ValueError(
                ".npz LSTM gate order {!r} != 'ifgo'; "
                "scripts/export_p3_student.py 로 다시 내보내세요".format(gate_order))
        if "obs_scale" in data and not int(data.get("obs_scale_folded", 0)):
            raise ValueError(
                "{}: obs_scale 이 lstm_weight_ih 에 접혀 있지 않습니다. 이 백엔드는 "
                "raw 관측을 그대로 먹이므로 스케일이 통째로 빠집니다.".format(path))
        self._lstm = {
            "wih": data["lstm_weight_ih"].astype(np.float32),
            "whh": data["lstm_weight_hh"].astype(np.float32),
            "bih": data["lstm_bias_ih"].astype(np.float32),
            "bhh": data["lstm_bias_hh"].astype(np.float32),
        }
        self._hidden = int(self._lstm["whh"].shape[1])
        self._in_dim = int(self._lstm["wih"].shape[1])
        bundle_obs = int(data["obs_dim"]) if "obs_dim" in data else None
        if bundle_obs is not None and bundle_obs != self._in_dim:
            raise ValueError(
                ".npz obs_dim {} 와 가중치 입력 {} 불일치 — 손상된 번들?".format(
                    bundle_obs, self._in_dim))
        self._backend = "numpy"

    def _load_onnx(self, path):
        import onnxruntime as ort

        self._sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._in_names = [i.name for i in self._sess.get_inputs()]
        self._out_names = [o.name for o in self._sess.get_outputs()]
        if len(self._in_names) < 3:
            raise ValueError("{}: (obs, h_in, c_in) 입력이 아닙니다 — recurrent "
                             "export 가 맞습니까?".format(path))
        self._in_dim = int(self._sess.get_inputs()[0].shape[-1])
        self._hidden = int(self._sess.get_inputs()[1].shape[-1])
        self._backend = "onnx"

    def _load_torchscript(self, path):
        import torch

        self._torch = torch
        self._module = torch.jit.load(path, map_location="cpu")
        self._module.eval()
        self._in_dim = int(self._module.obs_scale.numel())
        self._hidden = int(self._module.rnn.hidden_size)
        self._backend = "torchscript"

    # ---- 상태 ------------------------------------------------------------

    @property
    def hidden(self):
        """현재 LSTM hidden state (H,). 보조 헤드와 로깅에 씁니다."""
        return np.asarray(self._h, dtype=np.float32).reshape(-1)

    def reset(self):
        """LSTM hidden/cell 을 0 으로 — 에피소드 시작마다 필수.

        학습에서 에피소드가 hidden=0 에서 시작하므로, 기립 후 정책 인계 시점과
        중단 후 재개 시점에 호출하지 않으면 이전 세션의 기억이 남습니다.
        """
        if self._backend == "numpy":
            self._h = np.zeros(self._hidden, dtype=np.float32)
            self._c = np.zeros(self._hidden, dtype=np.float32)
        else:
            self._h = np.zeros((1, 1, self._hidden), dtype=np.float32)
            self._c = np.zeros((1, 1, self._hidden), dtype=np.float32)

    def estimate(self):
        """직전 스텝 latent 의 (부목 길이 L [m], base_lin_vel [m/s]). 헤드가 없으면 None."""
        if self.aux is None:
            return None
        return self.aux(self.hidden)

    # ---- 검증 ------------------------------------------------------------

    def _verify_reference_io(self):
        """모델 옆의 reference_io.json (개발 PC torch 출력)과 대조.

        파일이 없으면 조용히 통과. 있는데 안 맞으면 모터를 건드리기 전에 즉시
        중단합니다 — 잘못된/오염된 export 를 하드웨어에 올리는 것을 방지합니다.
        pairs 는 hidden=0 에서 시작하는 **하나의 에피소드**라 순서대로 실행하고,
        검증 후 상태를 다시 리셋합니다.
        """
        ref_path = os.path.join(self._dir, "reference_io.json")
        if not os.path.exists(ref_path):
            print("policy self-test SKIPPED: reference_io.json 이 없습니다")
            return
        with open(ref_path) as f:
            ref = json.load(f)
        self.reset()
        worst = 0.0
        for pair in ref["pairs"]:
            out = self(np.array(pair["obs"], dtype=np.float32))
            worst = max(worst, float(np.max(np.abs(out - np.array(pair["action"])))))
        self.reset()
        if worst > 1e-3:
            raise ValueError(
                "policy self-test MISMATCH: max|out-ref| = {:.3e} "
                "(reference: {})".format(worst, ref_path))
        print("policy self-test OK: {} pairs, max|out-ref| = {:.2e} [{}]".format(
            len(ref["pairs"]), worst, self._backend))

    # ---- 추론 ------------------------------------------------------------

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if x.shape[0] != C.OBS_DIM:
            raise ValueError("obs dim {} != {}".format(x.shape[0], C.OBS_DIM))

        if self._backend == "numpy":
            out = self._numpy_forward(x)
        elif self._backend == "onnx":
            out, self._h, self._c = self._sess.run(
                self._out_names,
                {self._in_names[0]: x.reshape(1, -1),
                 self._in_names[1]: self._h,
                 self._in_names[2]: self._c})
        else:
            with self._torch.no_grad():
                out, h, c = self._module(
                    self._torch.from_numpy(x.reshape(1, -1)),
                    self._torch.from_numpy(self._h),
                    self._torch.from_numpy(self._c))
            out, self._h, self._c = out.numpy(), h.numpy(), c.numpy()

        action = np.asarray(out, dtype=np.float64).reshape(-1)
        if action.shape[0] != C.NUM_ACTIONS:
            raise ValueError("action dim {} != {}".format(
                action.shape[0], C.NUM_ACTIONS))
        return action

    def _numpy_forward(self, x):
        """단일 스텝 LSTM (PyTorch gate 순서 i,f,g,o) + ELU MLP. h/c 를 제자리 갱신."""
        w = self._lstm
        H = self._hidden
        gates = w["wih"] @ x + w["bih"] + w["whh"] @ self._h + w["bhh"]
        i = _sigmoid(gates[:H])
        f = _sigmoid(gates[H:2 * H])
        g = np.tanh(gates[2 * H:3 * H])
        o = _sigmoid(gates[3 * H:])
        self._c = (f * self._c + i * g).astype(np.float32)
        self._h = (o * np.tanh(self._c)).astype(np.float32)

        h = self._h
        last = len(self._layers) - 1
        for k, (weight, bias) in enumerate(self._layers):
            h = h @ weight.T + bias
            if k < last:
                h = np.where(h > 0.0, h, np.exp(h) - 1.0)  # ELU
        return h

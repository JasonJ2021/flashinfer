import nvtx
import torch
from triton.testing import do_bench

import flashinfer


def normal_distribution(std):
    def normal_noise(shape, device):
        return torch.randn(shape, device=device) * std

    normal_noise.__name__ = f"normal_distribution(std={std})"
    return normal_noise


def gumbel_distribution(beta):
    def gumbel_noise(shape, device):
        U = torch.rand(shape, device=device)
        eps = 1e-20
        return torch.log(-torch.log(U + eps) + eps) / beta

    gumbel_noise.__name__ = f"gumbel_distribution(beta={beta})"
    return gumbel_noise


def init_seed_sampling(*args, **kwargs):
    torch.manual_seed(42)
    return flashinfer.sampling.sampling_from_probs(*args, **kwargs)


def init_seed_sampling_from_logits(*args, **kwargs):
    torch.manual_seed(42)
    return flashinfer.sampling.sampling_from_logits(*args, **kwargs)


def init_seed_sampling_from_softmax_logits(logits, *args, **kwargs):
    torch.manual_seed(42)
    return flashinfer.sampling.sampling_from_probs(
        torch.softmax(logits, dim=-1), *args, **kwargs
    )


def init_seed_top_k_sampling(*args, **kwargs):
    torch.manual_seed(42)
    return flashinfer.sampling.top_k_sampling_from_probs(*args, **kwargs)


def init_seed_top_p_sampling(*args, **kwargs):
    torch.manual_seed(42)
    return flashinfer.sampling.top_p_sampling_from_probs(*args, **kwargs)


def init_seed_radik_sampling(*args, **kwargs):
    torch.manual_seed(42)
    return flashinfer.sampling.radik_sampling_from_probs(*args, **kwargs)


@torch.inference_mode()
def bench_radik_profile():
    print("---")
    print("radik sampling")
    for vocab_size in [128512]:
        for batch_size in [64]:
            for distrib in [
                # normal_distribution(1),
                # normal_distribution(5),
                gumbel_distribution(0.1),
                # gumbel_distribution(1),
            ]:
                for k in [1000]:
                    logits = distrib((batch_size, vocab_size), device="cuda")
                    probs = torch.softmax(logits, dim=-1)
                    samples = torch.zeros(
                        batch_size, dtype=torch.int32, device=probs.device
                    )
                    ms = do_bench(
                        lambda: init_seed_radik_sampling(probs, k),
                        warmup=10,
                        rep=100,
                    )

                    io = (
                        probs.numel() * probs.element_size()
                        + samples.numel() * samples.element_size()
                    )
                    bandwidth = io * 1e-6 / ms
                    print(
                        f"vocab_size: {vocab_size}, batch_size: {batch_size}, distrib: {distrib.__name__}, k: {k}, duration: {ms * 1e3:.2f} us, effective bandwidth: {bandwidth:.2f} GB/s"
                    )


@torch.inference_mode()
def bench_radik():
    print("---")
    print("radik sampling")
    for vocab_size in [128512]:
        for batch_size in [1, 16, 32, 64, 128, 256, 512]:
            for distrib in [
                normal_distribution(1),
                normal_distribution(5),
                gumbel_distribution(0.1),
                gumbel_distribution(1),
            ]:
                for k in [10, 100, 1000]:
                    logits = distrib((batch_size, vocab_size), device="cuda")
                    probs = torch.softmax(logits, dim=-1)
                    samples = torch.zeros(
                        batch_size, dtype=torch.int32, device=probs.device
                    )
                    ms = do_bench(
                        lambda: init_seed_radik_sampling(probs, k),
                        warmup=100,
                        rep=1000,
                    )

                    io = (
                        probs.numel() * probs.element_size()
                        + samples.numel() * samples.element_size()
                    )
                    bandwidth = io * 1e-6 / ms
                    print(
                        f"vocab_size: {vocab_size}, batch_size: {batch_size}, distrib: {distrib.__name__}, k: {k}, duration: {ms * 1e3:.2f} us, effective bandwidth: {bandwidth:.2f} GB/s"
                    )


@torch.inference_mode()
def bench_radik_variable_k():
    print("---")
    print("radik variable k sampling")
    for vocab_size in [128512]:
        for batch_size in [1, 16, 32, 64, 128, 256, 512]:
            for distrib in [
                normal_distribution(1),
                normal_distribution(5),
                gumbel_distribution(0.1),
                gumbel_distribution(1),
            ]:
                for k in [10, 100, 1000]:
                    logits = distrib((batch_size, vocab_size), device="cuda")
                    probs = torch.softmax(logits, dim=-1)
                    samples = torch.zeros(
                        batch_size, dtype=torch.int32, device=probs.device
                    )
                    top_k_arr = torch.randint(1, k + 1, (batch_size,), device="cuda")
                    ms = do_bench(
                        lambda: init_seed_radik_sampling(probs, top_k_arr),
                        warmup=100,
                        rep=1000,
                    )

                    io = (
                        probs.numel() * probs.element_size()
                        + samples.numel() * samples.element_size()
                    )
                    bandwidth = io * 1e-6 / ms
                    print(
                        f"vocab_size: {vocab_size}, batch_size: {batch_size}, distrib: {distrib.__name__}, k: {k}, duration: {ms * 1e3:.2f} us, effective bandwidth: {bandwidth:.2f} GB/s"
                    )


@torch.inference_mode()
def bench_flashinfer_topk():
    print("---")
    print("top-k sampling")
    for vocab_size in [128512]:
        for batch_size in [1, 16, 32, 64, 128, 256, 512]:
            for distrib in [
                normal_distribution(1),
                normal_distribution(5),
                gumbel_distribution(0.1),
                gumbel_distribution(1),
            ]:
                for deterministic in [False]:
                    for k in [10, 100, 1000]:
                        logits = distrib((batch_size, vocab_size), device="cuda")
                        probs = torch.softmax(logits, dim=-1)
                        samples = torch.zeros(
                            batch_size, dtype=torch.int32, device=probs.device
                        )
                        ms = do_bench(
                            lambda: init_seed_top_k_sampling(
                                probs, k, deterministic=deterministic
                            ),
                            warmup=100,
                            rep=1000,
                        )

                        io = (
                            probs.numel() * probs.element_size()
                            + samples.numel() * samples.element_size()
                        )
                        bandwidth = io * 1e-6 / ms
                        print(
                            f"vocab_size: {vocab_size}, batch_size: {batch_size}, distrib: {distrib.__name__}, deterministic: {deterministic}, k: {k}, duration: {ms * 1e3:.2f} us, effective bandwidth: {bandwidth:.2f} GB/s"
                        )


@torch.inference_mode()
def bench_flashinfer_topk_variable_k():
    print("---")
    print("top-k variable k sampling")
    for vocab_size in [128512]:
        for batch_size in [1, 16, 32, 64, 128, 256, 512]:
            for distrib in [
                normal_distribution(1),
                normal_distribution(5),
                gumbel_distribution(0.1),
                gumbel_distribution(1),
            ]:
                for deterministic in [False]:
                    for k in [10, 100, 1000]:
                        logits = distrib((batch_size, vocab_size), device="cuda")
                        probs = torch.softmax(logits, dim=-1)
                        samples = torch.zeros(
                            batch_size, dtype=torch.int32, device=probs.device
                        )
                        top_k_arr = torch.randint(
                            1, k + 1, (batch_size,), device="cuda"
                        )
                        ms = do_bench(
                            lambda: init_seed_top_k_sampling(
                                probs, top_k_arr, deterministic=deterministic
                            ),
                            warmup=100,
                            rep=1000,
                        )

                        io = (
                            probs.numel() * probs.element_size()
                            + samples.numel() * samples.element_size()
                        )
                        bandwidth = io * 1e-6 / ms
                        print(
                            f"vocab_size: {vocab_size}, batch_size: {batch_size}, distrib: {distrib.__name__}, deterministic: {deterministic}, k: {k}, duration: {ms * 1e3:.2f} us, effective bandwidth: {bandwidth:.2f} GB/s"
                        )


if __name__ == "__main__":
    # bench_flashinfer_topk()
    # bench_flashinfer_topk_variable_k()
    # bench_radik()
    # bench_radik_variable_k()
    bench_radik_profile()

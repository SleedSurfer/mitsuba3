import numpy as np


def test_precision_drift(resolution):
    # Simulate a spiky region or a slope
    theta_vals = np.linspace(0, np.pi, resolution)

    # Pick a random bin index
    original_idx = np.random.randint(0, resolution - 1)
    # Pick a random offset inside the bin
    s = np.random.rand()

    # 1. Forward: Index -> Theta -> Vector
    angle_idx = float(original_idx) + s
    theta = angle_idx * (np.pi / (resolution - 1))

    # Create wo (assuming wi is [0,0,1])
    wo = np.array([np.sin(theta), 0, np.cos(theta)])

    # 2. Backward: Vector -> Theta -> Index
    # This simulates what eval_pdf() does
    reconstructed_cos_theta = wo[2]  # dot(wo, [0,0,1])
    reconstructed_theta = np.arccos(np.clip(reconstructed_cos_theta, -1.0, 1.0))

    reconstructed_idx_f = reconstructed_theta * ((resolution - 1) / np.pi)
    reconstructed_idx = int(np.floor(reconstructed_idx_f + 1e-9))  # common 'fix'

    if original_idx != reconstructed_idx:
        print(f"DRIFT DETECTED!")
        print(f"Original: {original_idx}")
        print(f"Reconstructed: {reconstructed_idx}")
        print(f"Diff: {reconstructed_idx_f - original_idx}")


# Run it 10k times


if __name__ == "__main__":
    for _ in range(10000):
      test_precision_drift(8192)
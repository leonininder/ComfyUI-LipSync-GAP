"""Frame-count contract for audio-driven output (round up to one video frame)."""
import math


def align_output_frames(frames, sample_count, sample_rate, frame_rate):
    if sample_count <= 0 or sample_rate <= 0 or frame_rate <= 0:
        raise ValueError('Audio duration and frame rate must be positive')
    expected = math.ceil(sample_count * frame_rate / sample_rate)
    if len(frames) < expected:
        raise RuntimeError(f'Inference returned {len(frames)} frames; expected at least {expected}')
    return frames[:expected]

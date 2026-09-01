from __future__ import annotations

import math

import numpy as np
import pytest

from daedalus.engine import (
    EstimateInputError,
    ShapeCalculationError,
    attention_parameter_count,
    batches_per_epoch,
    broadcast_output_shape,
    calculate_output_shapes,
    convolution_output_shape,
    convolution_parameter_count,
    estimate_attention_activation_memory,
    estimate_quantized_model_bytes,
    estimate_transformer_activation_memory,
    initialization_scale,
    matmul_output_shape,
    project_training_time,
    training_schedule,
    transformer_block_parameter_count,
)
from daedalus.layers import Embedding, Flatten, Sequential, TransformerBlock


def test_broadcast_and_matmul_shapes_match_hand_calculations() -> None:
    assert broadcast_output_shape((2, 1, 4), (1, 3, 4)) == (2, 3, 4)
    assert broadcast_output_shape((), (5, 1)) == (5, 1)
    assert matmul_output_shape((3,), (3,)) == ()
    assert matmul_output_shape((3,), (5, 3, 2)) == (5, 2)
    assert matmul_output_shape((5, 2, 3), (1, 3, 4)) == (5, 2, 4)


def test_shape_calculators_raise_educational_errors() -> None:
    with pytest.raises(ShapeCalculationError, match="conflict"):
        broadcast_output_shape((2, 3), (4, 3))
    with pytest.raises(ShapeCalculationError, match="contraction mismatch"):
        matmul_output_shape((2, 3), (4, 5))
    with pytest.raises(ShapeCalculationError, match="does not fit"):
        convolution_output_shape((2, 2), (5, 5))


def test_convolution_dimensions_and_parameters_match_manual_fixture() -> None:
    assert convolution_output_shape((32, 32), 3, stride=2, padding=1) == (16, 16)
    assert convolution_output_shape((20,), 3, dilation=2) == (16,)
    assert convolution_parameter_count(3, 16, (3, 3)) == 16 * 3 * 3 * 3 + 16
    assert convolution_parameter_count(8, 12, (3, 3), groups=4, bias=False) == 216


def test_attention_and_transformer_parameter_counts_match_implemented_layers() -> None:
    assert attention_parameter_count(8, 2) == 4 * 8 * 8 + 4 * 8 == 288
    expected_transformer = 288 + (8 * 32 + 32) + (32 * 8 + 8) + 4 * 8
    assert transformer_block_parameter_count(8, 2, 32) == expected_transformer == 872
    block = TransformerBlock(8, 2, feed_forward_dim=32, seed=1)
    assert sum(parameter.size for parameter in block.parameters()) == 872


def test_attention_and_transformer_activation_memory_components() -> None:
    attention = estimate_attention_activation_memory(2, 4, 8, 2, dtype=np.float32)
    assert attention.qkv_bytes == 3 * 2 * 4 * 8 * 4 == 768
    assert attention.score_bytes == attention.probability_bytes == 2 * 2 * 4 * 4 * 4
    assert attention.context_bytes == attention.output_bytes == 2 * 4 * 8 * 4
    assert attention.total_bytes == 1792

    transformer = estimate_transformer_activation_memory(2, 4, 8, 2, 32)
    assert transformer.attention_bytes == 1792
    assert transformer.normalization_bytes == transformer.residual_bytes == 512
    assert transformer.feed_forward_bytes == 2048
    assert transformer.total_bytes == 4864


def test_quantized_memory_includes_packing_and_group_metadata() -> None:
    estimate = estimate_quantized_model_bytes(
        100,
        bits_per_weight=4,
        group_size=25,
        zero_point_bits=4,
    )
    assert estimate.weight_bytes == 50
    assert estimate.scale_bytes == 16
    assert estimate.zero_point_bytes == 4
    assert estimate.total_bytes == 70
    assert estimate.compression_ratio_vs_float32 == pytest.approx(400 / 70)


def test_batches_schedule_and_training_time_projection() -> None:
    assert batches_per_epoch(101, 16) == 7
    assert batches_per_epoch(101, 16, drop_last=True) == 6
    schedule = training_schedule(101, 16, epochs=3, gradient_accumulation=2)
    assert schedule.batches_per_epoch == 7
    assert schedule.optimizer_steps_per_epoch == 4
    assert schedule.total_batches == 21
    assert schedule.total_optimizer_steps == 12

    projection = project_training_time(0.25, 4, 3, overhead_fraction=0.1)
    assert projection.seconds_per_epoch == pytest.approx(1.1)
    assert projection.total_seconds == pytest.approx(3.3)
    assert projection.total_minutes == pytest.approx(0.055)


def test_initialization_scales_match_standard_formulas() -> None:
    assert initialization_scale(8, 8) == pytest.approx(math.sqrt(6 / 16))
    assert initialization_scale(8, 8, method="xavier_normal") == pytest.approx(
        math.sqrt(2 / 16)
    )
    assert initialization_scale(8, method="he_normal") == pytest.approx(0.5)
    with pytest.raises(EstimateInputError, match="fan_out"):
        initialization_scale(8, method="xavier_uniform")


def test_existing_shape_pipeline_understands_embedding_transformer_and_flatten() -> None:
    model = Sequential(
        Embedding(20, 8, seed=2),
        TransformerBlock(8, 2, feed_forward_dim=16, seed=3),
        Flatten(start_dim=1),
    )
    shapes = calculate_output_shapes(model, (2, 5))
    assert [item.output_shape for item in shapes] == [(2, 5, 8), (2, 5, 8), (2, 40)]


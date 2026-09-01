"""Reusable native panel for Daedalus' advanced engineering calculators."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from daedalus.engine import (
    CalculatorError,
    ShapeCalculationError,
    attention_parameter_count,
    convolution_output_shape,
    convolution_parameter_count,
    estimate_attention_activation_memory,
    estimate_quantized_model_bytes,
    estimate_transformer_activation_memory,
    project_training_time,
    training_schedule,
    transformer_block_parameter_count,
)
from daedalus.gui.widgets import Card, human_bytes


class AdvancedCalculatorPanel(QWidget):
    """Tabbed, self-contained UI for advanced architecture estimates.

    Integration is intentionally one line after construction:
    ``layout.addWidget(AdvancedCalculatorPanel(parent))``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AdvancedCalculatorPanel")
        self.setAccessibleName("Advanced engineering calculators")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        title = QLabel("Advanced Engineering Calculators")
        title.setObjectName("PageTitle")
        title.setAccessibleName("Advanced Engineering Calculators title")
        root.addWidget(title)
        subtitle = QLabel(
            "Plan convolution geometry, attention capacity, quantized storage, and bounded "
            "training runs without sending data outside this computer."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.tabs.setAccessibleName("Advanced calculator categories")
        self.tabs.addTab(self._scroll(self._build_convolution_tab()), "Convolution")
        self.tabs.addTab(self._scroll(self._build_attention_tab()), "Attention")
        self.tabs.addTab(self._scroll(self._build_quantization_tab()), "Quantization")
        self.tabs.addTab(self._scroll(self._build_training_tab()), "Training plan")
        self.tabs.setTabToolTip(0, "Convolution output geometry and trainable parameters")
        self.tabs.setTabToolTip(1, "Attention and Transformer parameters and activation memory")
        self.tabs.setTabToolTip(2, "Packed model storage and compression savings")
        self.tabs.setTabToolTip(3, "Batches, optimizer steps, and projected wall time")
        root.addWidget(self.tabs, 1)

        self.recalculate_all()

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidgetResizable(True)
        area.setWidget(widget)
        widget.setAutoFillBackground(False)
        return area

    @staticmethod
    def _page() -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        return page, layout

    @staticmethod
    def _form(card: Card) -> QFormLayout:
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)
        card.body.addLayout(form)
        return form

    @staticmethod
    def _line(value: str, accessible_name: str, placeholder: str = "") -> QLineEdit:
        field = QLineEdit(value)
        field.setAccessibleName(accessible_name)
        field.setClearButtonEnabled(True)
        field.setPlaceholderText(placeholder)
        return field

    @staticmethod
    def _spin(
        value: int,
        minimum: int,
        maximum: int,
        accessible_name: str,
    ) -> QSpinBox:
        field = QSpinBox()
        field.setRange(minimum, maximum)
        field.setValue(value)
        field.setKeyboardTracking(False)
        field.setAccessibleName(accessible_name)
        return field

    @staticmethod
    def _result_card(title: str, accessible_name: str) -> tuple[Card, QLabel]:
        card = Card(title, "Results remain estimates; validate against measured workloads.")
        result = QLabel("Ready to calculate.")
        result.setObjectName("Muted")
        result.setWordWrap(True)
        result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        result.setAccessibleName(accessible_name)
        card.body.addWidget(result)
        return card, result

    @staticmethod
    def _button(label: str, accessible_name: str, callback: Callable[[], None]) -> QPushButton:
        button = QPushButton(label)
        button.setObjectName("Primary")
        button.setAccessibleName(accessible_name)
        button.clicked.connect(callback)
        return button

    def _build_convolution_tab(self) -> QWidget:
        page, layout = self._page()
        inputs = Card(
            "Convolution geometry",
            "Use one value for every spatial axis or comma-separated values such as 3, 5.",
            accent=True,
        )
        form = self._form(inputs)
        self.convolution_spatial = self._line(
            "32, 32",
            "Convolution input spatial dimensions",
            "height, width",
        )
        self.convolution_kernel = self._line("3", "Convolution kernel dimensions", "3 or 3, 5")
        self.convolution_stride = self._line("2", "Convolution stride dimensions", "1")
        self.convolution_padding = self._line("1", "Convolution padding dimensions", "0")
        self.convolution_dilation = self._line("1", "Convolution dilation dimensions", "1")
        self.convolution_in_channels = self._spin(3, 1, 1_000_000, "Convolution input channels")
        self.convolution_out_channels = self._spin(
            16,
            1,
            1_000_000,
            "Convolution output channels",
        )
        self.convolution_groups = self._spin(1, 1, 1_000_000, "Convolution groups")
        self.convolution_bias = QCheckBox("Include one bias per output channel")
        self.convolution_bias.setChecked(True)
        self.convolution_bias.setAccessibleName("Include convolution bias parameters")
        form.addRow("Input spatial shape", self.convolution_spatial)
        form.addRow("Kernel", self.convolution_kernel)
        form.addRow("Stride", self.convolution_stride)
        form.addRow("Padding", self.convolution_padding)
        form.addRow("Dilation", self.convolution_dilation)
        form.addRow("Input channels", self.convolution_in_channels)
        form.addRow("Output channels", self.convolution_out_channels)
        form.addRow("Groups", self.convolution_groups)
        form.addRow("Bias", self.convolution_bias)
        inputs.body.addWidget(
            self._button(
                "Calculate convolution",
                "Calculate convolution geometry and parameters",
                self.calculate_convolution,
            )
        )
        layout.addWidget(inputs)
        result_card, self.convolution_result = self._result_card(
            "Convolution result",
            "Convolution calculation result",
        )
        layout.addWidget(result_card)
        layout.addStretch(1)
        return page

    def _build_attention_tab(self) -> QWidget:
        page, layout = self._page()
        inputs = Card(
            "Attention and Transformer",
            "Dense pre-norm Transformer block with separate Q, K, V, and output projections.",
            accent=True,
        )
        form = self._form(inputs)
        self.attention_batch = self._spin(2, 1, 1_000_000, "Attention batch size")
        self.attention_sequence = self._spin(128, 1, 1_000_000, "Attention sequence length")
        self.attention_width = self._spin(256, 1, 1_000_000, "Attention embedding width")
        self.attention_heads = self._spin(8, 1, 100_000, "Attention head count")
        self.attention_feed_forward = self._spin(
            1024,
            1,
            10_000_000,
            "Transformer feed-forward width",
        )
        self.attention_bias = QCheckBox("Include projection and feed-forward biases")
        self.attention_bias.setChecked(True)
        self.attention_bias.setAccessibleName("Include Transformer bias parameters")
        self.attention_dtype = QComboBox()
        self.attention_dtype.addItem("Float16 · 2 bytes", "float16")
        self.attention_dtype.addItem("Float32 · 4 bytes", "float32")
        self.attention_dtype.addItem("Float64 · 8 bytes", "float64")
        self.attention_dtype.setCurrentIndex(1)
        self.attention_dtype.setAccessibleName("Attention activation precision")
        form.addRow("Batch size", self.attention_batch)
        form.addRow("Sequence length", self.attention_sequence)
        form.addRow("Embedding width", self.attention_width)
        form.addRow("Attention heads", self.attention_heads)
        form.addRow("Feed-forward width", self.attention_feed_forward)
        form.addRow("Activation precision", self.attention_dtype)
        form.addRow("Bias", self.attention_bias)
        inputs.body.addWidget(
            self._button(
                "Estimate attention",
                "Calculate attention and Transformer estimates",
                self.calculate_attention,
            )
        )
        layout.addWidget(inputs)
        result_card, self.attention_result = self._result_card(
            "Attention result",
            "Attention and Transformer calculation result",
        )
        layout.addWidget(result_card)
        layout.addStretch(1)
        return page

    def _build_quantization_tab(self) -> QWidget:
        page, layout = self._page()
        inputs = Card(
            "Quantized model storage",
            "Packed weight bytes plus optional per-group scale and zero-point metadata.",
            accent=True,
        )
        form = self._form(inputs)
        self.quantization_parameters = self._spin(
            100_000_000,
            1,
            2_000_000_000,
            "Quantized model parameter count",
        )
        self.quantization_bits = QComboBox()
        for bits in (2, 3, 4, 8, 16):
            self.quantization_bits.addItem(f"{bits}-bit weights", bits)
        self.quantization_bits.setCurrentIndex(2)
        self.quantization_bits.setAccessibleName("Quantized bits per weight")
        self.quantization_group = self._spin(
            64,
            0,
            1_000_000,
            "Quantization group size; zero disables grouping",
        )
        self.quantization_group.setSpecialValueText("No groups")
        self.quantization_scale_dtype = QComboBox()
        self.quantization_scale_dtype.addItem("Float16 scales", "float16")
        self.quantization_scale_dtype.addItem("Float32 scales", "float32")
        self.quantization_scale_dtype.setCurrentIndex(1)
        self.quantization_scale_dtype.setAccessibleName("Quantization scale precision")
        self.quantization_zero_bits = QComboBox()
        for bits in (0, 4, 8, 16):
            label = "No zero point" if bits == 0 else f"{bits}-bit zero point"
            self.quantization_zero_bits.addItem(label, bits)
        self.quantization_zero_bits.setAccessibleName("Quantization zero point precision")
        form.addRow("Parameters", self.quantization_parameters)
        form.addRow("Weight precision", self.quantization_bits)
        form.addRow("Group size", self.quantization_group)
        form.addRow("Scale precision", self.quantization_scale_dtype)
        form.addRow("Zero point", self.quantization_zero_bits)
        inputs.body.addWidget(
            self._button(
                "Estimate storage",
                "Calculate quantized model storage",
                self.calculate_quantization,
            )
        )
        layout.addWidget(inputs)
        result_card, self.quantization_result = self._result_card(
            "Quantization result",
            "Quantized storage calculation result",
        )
        layout.addWidget(result_card)
        layout.addStretch(1)
        return page

    def _build_training_tab(self) -> QWidget:
        page, layout = self._page()
        inputs = Card(
            "Batch and time planner",
            "Step latency means measured time per optimizer update, not per micro-batch.",
            accent=True,
        )
        form = self._form(inputs)
        self.training_samples = self._spin(50_000, 1, 2_000_000_000, "Training sample count")
        self.training_batch = self._spin(64, 1, 10_000_000, "Training batch size")
        self.training_epochs = self._spin(10, 1, 1_000_000, "Training epoch count")
        self.training_accumulation = self._spin(
            1,
            1,
            1_000_000,
            "Gradient accumulation batches per optimizer step",
        )
        self.training_drop_last = QCheckBox("Discard the final incomplete batch")
        self.training_drop_last.setAccessibleName("Drop incomplete final training batch")
        self.training_step_seconds = QDoubleSpinBox()
        self.training_step_seconds.setRange(0.000001, 1_000_000.0)
        self.training_step_seconds.setDecimals(6)
        self.training_step_seconds.setValue(0.125)
        self.training_step_seconds.setSuffix(" s")
        self.training_step_seconds.setAccessibleName("Seconds per optimizer step")
        self.training_overhead = QDoubleSpinBox()
        self.training_overhead.setRange(0.0, 1_000.0)
        self.training_overhead.setDecimals(1)
        self.training_overhead.setValue(10.0)
        self.training_overhead.setSuffix(" %")
        self.training_overhead.setAccessibleName("Training overhead percentage")
        form.addRow("Samples", self.training_samples)
        form.addRow("Batch size", self.training_batch)
        form.addRow("Epochs", self.training_epochs)
        form.addRow("Gradient accumulation", self.training_accumulation)
        form.addRow("Remainder", self.training_drop_last)
        form.addRow("Seconds per optimizer step", self.training_step_seconds)
        form.addRow("I/O and validation overhead", self.training_overhead)
        inputs.body.addWidget(
            self._button(
                "Project training run",
                "Calculate batches steps and projected training time",
                self.calculate_training_plan,
            )
        )
        layout.addWidget(inputs)
        result_card, self.training_result = self._result_card(
            "Training projection",
            "Training schedule and time calculation result",
        )
        layout.addWidget(result_card)
        layout.addStretch(1)
        return page

    @staticmethod
    def _parse_shape(text: str, name: str) -> tuple[int, ...]:
        pieces = [piece for piece in re.split(r"[xX×,;\s]+", text.strip()) if piece]
        if not pieces:
            raise ShapeCalculationError(f"{name} cannot be empty")
        try:
            return tuple(int(piece) for piece in pieces)
        except ValueError as exc:
            raise ShapeCalculationError(f"{name} must contain whole-number dimensions") from exc

    @classmethod
    def _parse_dimension_value(cls, text: str, name: str) -> int | tuple[int, ...]:
        values = cls._parse_shape(text, name)
        return values[0] if len(values) == 1 else values

    @staticmethod
    def _duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        if seconds < 3600:
            return f"{seconds / 60:.2f} minutes"
        if seconds < 86_400:
            return f"{seconds / 3600:.2f} hours"
        return f"{seconds / 86_400:.2f} days"

    @staticmethod
    def _set_result(label: QLabel, text: str, *, success: bool) -> None:
        label.setText(text)
        label.setObjectName("Success" if success else "Danger")
        label.setAccessibleDescription("Calculation succeeded" if success else "Calculation failed")
        style = label.style()
        style.unpolish(label)
        style.polish(label)

    def _run_safely(self, label: QLabel, operation: Callable[[], str]) -> None:
        try:
            self._set_result(label, operation(), success=True)
        except (CalculatorError, ValueError, OverflowError) as exc:
            self._set_result(label, f"Check input: {exc}", success=False)
        except Exception:
            self._set_result(
                label,
                "Calculation stopped safely. Review the inputs and try a smaller bounded estimate.",
                success=False,
            )

    def calculate_convolution(self) -> None:
        def operation() -> str:
            spatial = self._parse_shape(self.convolution_spatial.text(), "input spatial shape")
            kernel = self._parse_dimension_value(self.convolution_kernel.text(), "kernel")
            stride = self._parse_dimension_value(self.convolution_stride.text(), "stride")
            padding = self._parse_dimension_value(self.convolution_padding.text(), "padding")
            dilation = self._parse_dimension_value(self.convolution_dilation.text(), "dilation")
            output = convolution_output_shape(
                spatial,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            parameter_kernel = (kernel,) * len(spatial) if isinstance(kernel, int) else kernel
            parameters = convolution_parameter_count(
                self.convolution_in_channels.value(),
                self.convolution_out_channels.value(),
                parameter_kernel,
                groups=self.convolution_groups.value(),
                bias=self.convolution_bias.isChecked(),
            )
            geometry = " × ".join(str(dimension) for dimension in output)
            return (
                f"Output spatial shape: {geometry}\n"
                f"Trainable parameters: {parameters:,}\n"
                "Formula per axis: floor((input + 2·padding − dilation·(kernel−1) − 1) "
                "/ stride) + 1"
            )

        self._run_safely(self.convolution_result, operation)

    def calculate_attention(self) -> None:
        def operation() -> str:
            batch = self.attention_batch.value()
            sequence = self.attention_sequence.value()
            width = self.attention_width.value()
            heads = self.attention_heads.value()
            feed_forward = self.attention_feed_forward.value()
            bias = self.attention_bias.isChecked()
            dtype = str(self.attention_dtype.currentData())
            attention_parameters = attention_parameter_count(width, heads, bias=bias)
            block_parameters = transformer_block_parameter_count(
                width,
                heads,
                feed_forward,
                bias=bias,
            )
            attention_memory = estimate_attention_activation_memory(
                batch,
                sequence,
                width,
                heads,
                dtype=dtype,
            )
            block_memory = estimate_transformer_activation_memory(
                batch,
                sequence,
                width,
                heads,
                feed_forward,
                dtype=dtype,
            )
            parameter_bytes = block_parameters * np.dtype(dtype).itemsize
            return (
                f"Attention parameters: {attention_parameters:,}\n"
                f"Transformer block parameters: {block_parameters:,} "
                f"({human_bytes(parameter_bytes)} at {dtype})\n"
                f"Attention activations: {human_bytes(attention_memory.total_bytes)}\n"
                f"Transformer activations: {human_bytes(block_memory.total_bytes)}\n"
                f"Attention score + probability matrices: "
                f"{human_bytes(attention_memory.score_bytes + attention_memory.probability_bytes)}"
            )

        self._run_safely(self.attention_result, operation)

    def calculate_quantization(self) -> None:
        def operation() -> str:
            parameters = self.quantization_parameters.value()
            group_value = self.quantization_group.value()
            estimate = estimate_quantized_model_bytes(
                parameters,
                bits_per_weight=int(self.quantization_bits.currentData()),
                group_size=group_value or None,
                scale_dtype=str(self.quantization_scale_dtype.currentData()),
                zero_point_bits=int(self.quantization_zero_bits.currentData()),
            )
            baseline = parameters * np.dtype(np.float32).itemsize
            saved = baseline - estimate.total_bytes
            savings_percent = 100.0 * saved / baseline
            return (
                f"Packed total: {human_bytes(estimate.total_bytes)}\n"
                f"Weights: {human_bytes(estimate.weight_bytes)} · "
                f"scales: {human_bytes(estimate.scale_bytes)} · "
                f"zero points: {human_bytes(estimate.zero_point_bytes)}\n"
                f"FP32 baseline: {human_bytes(baseline)}\n"
                f"Projected savings: {human_bytes(saved)} ({savings_percent:.1f}%) · "
                f"{estimate.compression_ratio_vs_float32:.2f}× smaller"
            )

        self._run_safely(self.quantization_result, operation)

    def calculate_training_plan(self) -> None:
        def operation() -> str:
            schedule = training_schedule(
                self.training_samples.value(),
                self.training_batch.value(),
                epochs=self.training_epochs.value(),
                gradient_accumulation=self.training_accumulation.value(),
                drop_last=self.training_drop_last.isChecked(),
            )
            timing = project_training_time(
                self.training_step_seconds.value(),
                schedule.optimizer_steps_per_epoch,
                self.training_epochs.value(),
                overhead_fraction=self.training_overhead.value() / 100.0,
            )
            return (
                f"Batches per epoch: {schedule.batches_per_epoch:,}\n"
                f"Optimizer steps per epoch: {schedule.optimizer_steps_per_epoch:,}\n"
                f"Total batches: {schedule.total_batches:,} · "
                f"total optimizer steps: {schedule.total_optimizer_steps:,}\n"
                f"Projected epoch time: {self._duration(timing.seconds_per_epoch)}\n"
                f"Projected run time: {self._duration(timing.total_seconds)}"
            )

        self._run_safely(self.training_result, operation)

    def recalculate_all(self) -> None:
        """Refresh every tab, useful after loading saved UI settings."""

        self.calculate_convolution()
        self.calculate_attention()
        self.calculate_quantization()
        self.calculate_training_plan()


__all__ = ["AdvancedCalculatorPanel"]

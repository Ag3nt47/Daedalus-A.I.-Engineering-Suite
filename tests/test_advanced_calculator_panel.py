"""Offscreen smoke tests for the isolated advanced calculator panel."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from daedalus.gui.advanced_calculator_panel import AdvancedCalculatorPanel


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication(["daedalus-advanced-calculator-test"])


def test_panel_builds_accessible_tabs_and_default_results(app: QApplication) -> None:
    panel = AdvancedCalculatorPanel()
    panel.show()
    app.processEvents()
    try:
        assert panel.accessibleName() == "Advanced engineering calculators"
        assert [panel.tabs.tabText(index) for index in range(panel.tabs.count())] == [
            "Convolution",
            "Attention",
            "Quantization",
            "Training plan",
        ]
        assert "Output spatial shape: 16 × 16" in panel.convolution_result.text()
        assert "Transformer block parameters" in panel.attention_result.text()
        assert "Projected savings" in panel.quantization_result.text()
        assert "Optimizer steps per epoch" in panel.training_result.text()
        assert panel.convolution_spatial.accessibleName()
        assert panel.attention_width.accessibleName()
        assert panel.quantization_parameters.accessibleName()
        assert panel.training_step_seconds.accessibleName()
    finally:
        panel.close()
        panel.deleteLater()
        app.processEvents()


def test_invalid_input_fails_closed_and_recovers(app: QApplication) -> None:
    panel = AdvancedCalculatorPanel()
    try:
        panel.convolution_kernel.setText("not-a-dimension")
        panel.calculate_convolution()
        assert panel.convolution_result.objectName() == "Danger"
        assert panel.convolution_result.text().startswith("Check input:")

        panel.convolution_kernel.setText("3")
        panel.calculate_convolution()
        assert panel.convolution_result.objectName() == "Success"
        assert "Trainable parameters: 448" in panel.convolution_result.text()

        panel.attention_width.setValue(10)
        panel.attention_heads.setValue(3)
        panel.calculate_attention()
        assert panel.attention_result.objectName() == "Danger"
        assert "divisible" in panel.attention_result.text()
    finally:
        panel.deleteLater()
        app.processEvents()


def test_known_quantization_and_training_fixtures(app: QApplication) -> None:
    panel = AdvancedCalculatorPanel()
    try:
        panel.quantization_parameters.setValue(100)
        panel.quantization_bits.setCurrentIndex(panel.quantization_bits.findData(4))
        panel.quantization_group.setValue(25)
        panel.quantization_zero_bits.setCurrentIndex(panel.quantization_zero_bits.findData(4))
        panel.calculate_quantization()
        assert "Packed total: 70.0 B" in panel.quantization_result.text()
        assert "82.5%" in panel.quantization_result.text()

        panel.training_samples.setValue(101)
        panel.training_batch.setValue(16)
        panel.training_epochs.setValue(3)
        panel.training_accumulation.setValue(2)
        panel.training_step_seconds.setValue(0.25)
        panel.training_overhead.setValue(10.0)
        panel.calculate_training_plan()
        assert "Batches per epoch: 7" in panel.training_result.text()
        assert "Optimizer steps per epoch: 4" in panel.training_result.text()
        assert "Total batches: 21" in panel.training_result.text()
        assert "Projected run time: 3.30 seconds" in panel.training_result.text()
    finally:
        panel.deleteLater()
        app.processEvents()


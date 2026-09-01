"""Train a small from-scratch network on deterministic XOR data."""

from __future__ import annotations

import numpy as np

from daedalus.engine import Trainer, make_xor
from daedalus.layers import Linear, Sequential, Tanh
from daedalus.losses import CrossEntropyLoss
from daedalus.optim import Adam


def main() -> None:
    features, targets = make_xor(n_samples=240, noise=0.08, seed=7)
    model = Sequential(
        Linear(2, 8, seed=11),
        Tanh(),
        Linear(8, 2, seed=12),
    )
    trainer = Trainer(
        model,
        CrossEntropyLoss(),
        Adam(model.parameters(), lr=0.03),
        seed=13,
    )
    history = trainer.fit(features, targets, epochs=100, batch_size=24)
    predictions = trainer.predict(features).argmax(axis=1)
    accuracy = float(np.mean(predictions == targets))
    print(f"initial loss: {history.loss[0]:.6f}")
    print(f"final loss:   {history.loss[-1]:.6f}")
    print(f"accuracy:     {accuracy:.1%}")


if __name__ == "__main__":
    main()


"""Recover a deterministic synthetic linear relationship."""

from __future__ import annotations

from daedalus.engine import Trainer, make_regression
from daedalus.layers import Linear, Sequential
from daedalus.losses import MSELoss
from daedalus.optim import Adam


def main() -> None:
    features, targets, true_weight, true_bias = make_regression(
        n_samples=200,
        n_features=3,
        noise=0.02,
        seed=21,
        return_parameters=True,
    )
    model = Sequential(Linear(3, 1, seed=22))
    trainer = Trainer(model, MSELoss(), Adam(model.parameters(), lr=0.04), seed=23)
    history = trainer.fit(features, targets, epochs=120, batch_size=20)
    learned = model.layers[0]
    print(f"initial loss: {history.loss[0]:.6f}")
    print(f"final loss:   {history.loss[-1]:.6f}")
    print("true weight:", true_weight.ravel())
    print("fit weight: ", learned.weight.data.ravel())
    print("true bias:  ", true_bias)
    print("fit bias:   ", learned.bias.data if learned.bias is not None else None)


if __name__ == "__main__":
    main()


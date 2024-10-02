"""fraud detection main file."""

from typing import Annotated

from typer import Typer, Argument


import tasks.train
import tasks.prepare
import tasks.evaluate


app = Typer()


@app.command()
def prepare(cache_path: Annotated[str, Argument(envvar="CACHE_PATH")]) -> None:
    tasks.prepare.run(
        cache_path=cache_path,
    )


@app.command()
def train(
    cache_path: Annotated[str, Argument(envvar="CACHE_PATH")],
    model_name: Annotated[str, Argument(envvar="MODEL_NAME")],
) -> None:
    tasks.train.run(cache_path=cache_path, model=model_name)


@app.command()
def evaluate(
    cache_path: Annotated[str, Argument(envvar="CACHE_PATH")],
    model_name: Annotated[str, Argument(envvar="MODEL_NAME")],
) -> None:
    tasks.evaluate.run(cache_path=cache_path, model=model_name)


if __name__ == "__main__":
    app()

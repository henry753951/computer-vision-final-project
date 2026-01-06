import typer
from typing_extensions import Annotated
from core.manager import CLIManager
from src.dataloader import ImageLoader
from utils.logger import logger

app = typer.Typer(
    name="CV Training CLI",
    add_completion=False,
    help="CLI for training CNN models with strict data caching.",
)
manager = CLIManager()


@app.command("list-models")
def list_models():
    """List all supported CNN architectures."""
    manager.list_models()


@app.command("train")
def train(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Name of the model (e.g., VGG16)")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force rebuild dataset cache")
    ] = False,
):
    """Train a specific model using cached metadata if available."""
    if not model:
        for m in manager.get_supported_models():
            manager.train_model(m, force_refresh=force)
    else:
        manager.train_model(model, force_refresh=force)


@app.command("benchmark")
def benchmark(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Name of the model to benchmark")
    ] = None,
):
    if model:
        """Benchmark a specific model."""
        manager.benchmark_model(model)
    else:
        """Benchmark all supported models."""
        manager.benchmark_all_models()


@app.command("dataset")
def info_dataset(
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force rebuild dataset cache")
    ] = False,
):
    """Display dataset information and force cache rebuild if specified."""

    dataset = ImageLoader.load_dataset(force_refresh=force)
    logger.info(
        f"Dataset Info - Train Steps: {dataset.train_steps}, Val Steps: {dataset.val_steps}, Classes: {dataset.num_classes}"
    )

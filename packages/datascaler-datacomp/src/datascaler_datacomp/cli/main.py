import typer

from ._evaluate import app as app_evaluate
from ._train import app as app_train

app = typer.Typer()

app.add_typer(app_train)
app.add_typer(app_evaluate)

if __name__ == "__main__":
    app(prog_name="datascaler-datacomp")

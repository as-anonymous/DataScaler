import typer

from ._train import app as app_train

app = typer.Typer()

app.add_typer(app_train)

if __name__ == "__main__":
    app(prog_name="datascaler-open-clip")

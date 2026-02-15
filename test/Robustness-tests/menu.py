from InquirerPy import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
import time
from pathlib import Path

from image_robustness import ImageRobustnessTool

console = Console()


# ------------------------------------------------
def mostrar_bienvenida():
    console.print(
        Panel.fit(
            "[bold cyan]Herramienta de modificación de imágenes[/bold cyan]\n"
            "Evaluación de robustez del modelo",
            border_style="green",
        )
    )


# ------------------------------------------------
def barra_inicio():

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
    ) as progress:

        t1 = progress.add_task("Cargando dataset...", total=100)
        for _ in range(100):
            time.sleep(0.01)
            progress.update(t1, advance=1)

        t2 = progress.add_task("Verificando folder salida...", total=100)
        for _ in range(100):
            time.sleep(0.01)
            progress.update(t2, advance=1)


# ------------------------------------------------
def menu():
    return inquirer.select(
        message="Selecciona una transformación:",
        choices=[
            "Todas",
            "Variación de iluminación",
            "Ruido gaussiano",
            "Desplazamiento de ROI",
            "Blur",
            "Burbujas",
            "Ruido sal y pimienta",
            "Salir",
        ],
    ).execute()


# ------------------------------------------------
def main():

    mostrar_bienvenida()
    barra_inicio()

    # 🔥 directorio REAL donde está menu.py
    BASE_DIR = Path(__file__).resolve().parent

    # rutas absolutas robustas
    photos_dir = (BASE_DIR / "../../data").resolve()
    dataset_csv = (BASE_DIR / "../../data/database/labeled-dataset.csv").resolve()
    output_dir = (BASE_DIR / "generated-datasets").resolve()

    console.print(f"[cyan]Dataset:[/cyan] {dataset_csv}")
    console.print(f"[cyan]Photos:[/cyan] {photos_dir}")
    console.print(f"[cyan]Output:[/cyan] {output_dir}")

    tool = ImageRobustnessTool(
        photos_dir=photos_dir,
        dataset_csv=dataset_csv,
        output_dir=output_dir,
    )

    mapping = {
        "Variación de iluminación": "variacion_iluminacion",
        "Ruido gaussiano": "ruido_gaussiano",
        "Desplazamiento de ROI": "desplazamiento_roi",
        "Blur": "blur",
        "Burbujas": "burbujas",
        "Ruido sal y pimienta": "ruido_sal_pimienta",
    }

    while True:

        opcion = menu()

        if opcion == "Salir":
            console.print("[bold red]Saliendo...[/bold red]")
            break

        elif opcion == "Todas":
            tool.ejecutar_todo()

        else:
            tool.procesar_modificacion(mapping[opcion])

if __name__ == "__main__":
    main()

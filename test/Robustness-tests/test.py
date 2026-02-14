from InquirerPy import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
import time
import os

console = Console()

# -----------------------------
# Configuración
# -----------------------------
DATASET_PATH = "./dataset"
OUTPUT_PATH = "./output"


# -----------------------------
# Funciones de inicialización
# -----------------------------
def mostrar_bienvenida():
    console.print(
        Panel.fit(
            "[bold cyan]Bienvenido a la herramienta de modificación de imágenes[/bold cyan]\n\n"
            "Esta herramienta permite generar perturbaciones controladas para\n"
            "evaluar la robustez de este modelo.\n\n"
            "[yellow]Elige una opción para modificar las imágenes.[/yellow]",
            title="Image Robustness Tool",
            border_style="green",
        )
    )


def inicializar_sistema():
    """Simula carga del dataset y validación de carpetas"""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=True,
    ) as progress:

        task1 = progress.add_task("Cargando dataset...", total=100)
        for _ in range(100):
            time.sleep(0.01)
            progress.update(task1, advance=1)

        task2 = progress.add_task("Verificando folder de salida...", total=100)
        for _ in range(100):
            time.sleep(0.01)
            progress.update(task2, advance=1)

    # verificación real
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        console.print("[green]✔ Folder de salida creado[/green]")
    else:
        console.print("[green]✔ Folder de salida verificado[/green]")


# -----------------------------
# Acciones (stubs)
# -----------------------------
def todo():
    console.print("[cyan]Aplicando transformación...[/cyan]")

def variacion_iluminacion():
    console.print("[cyan]Aplicando variación de iluminación...[/cyan]")


def ruido_gaussiano():
    console.print("[cyan]Aplicando ruido gaussiano...[/cyan]")


def desplazamiento_roi():
    console.print("[cyan]Aplicando desplazamiento de ROI...[/cyan]")


def blur():
    console.print("[cyan]Aplicando blur...[/cyan]")


def burbujas():
    console.print("[cyan]Generando burbujas...[/cyan]")


def ruido_sal_pimienta():
    console.print("[cyan]Aplicando ruido sal y pimienta...[/cyan]")


# -----------------------------
# Menú principal
# -----------------------------
def menu():
    opcion = inquirer.select(
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

    return opcion


# -----------------------------
# Loop principal
# -----------------------------
def main():
    mostrar_bienvenida()
    inicializar_sistema()

    while True:
        opcion = menu()

        if opcion == "Variación de iluminación":
            variacion_iluminacion()

        elif opcion == "Ruido gaussiano":
            ruido_gaussiano()

        elif opcion == "Desplazamiento de ROI":
            desplazamiento_roi()

        elif opcion == "Blur":
            blur()

        elif opcion == "Burbujas":
            burbujas()

        elif opcion == "Ruido sal y pimienta":
            ruido_sal_pimienta()

        elif opcion == "Salir":
            console.print("[bold red]Saliendo...[/bold red]")
            break


if __name__ == "__main__":
    main()

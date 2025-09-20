from typing import Union

from rich.table import Table
from rich.console import Console

CONSOLE = Console()


class DataStatPrinter:
    def __init__(self):
        self.data = {"num_train": 0, "num_eval": 0}

    def generate_table(self) -> Table:
        table = Table()
        table.add_column("Num Train", style="bold")
        table.add_column("Num Eval", style="bold")

        table.add_row(
            f"{self.data['num_train']}",
            f"{self.data['num_eval']}",
        )

        return table

    def print_table(self):
        table = self.generate_table()
        CONSOLE.print(table)

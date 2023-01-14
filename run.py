import tasks
import constants
import numpy as np
import click


@click.command()
@click.argument("todos", nargs=-1)
def main(todos):
    _print_hello_message()
    for task in _get_tasks(todos):
        task.run("output")


def _get_tasks(todos):
    tasks_to_do = [
        tasks.PointNucleus(
            name="PointNucleus (Task 1)",
            rmin=1e-15 * constants.A_BHOR,
            rmax=10 * constants.A_BHOR,
            n_grid_points=int(1e3 + 1),
            energies=[-(0.9 + i * 0.05) * constants.RY for i in range(0, 5)],
        ),
        tasks.PointNucleusFindBoundState(
            name="PointNucleusFindBoundState (Task 2)",
            energy_min=-1.2 * constants.RY,
            energy_max=-0.8 * constants.RY,
            rmin=1e-15 * constants.A_BHOR,
            max_radii=np.array([5, 10, 15, 20]) * constants.A_BHOR,
            l_level=0,
            numbers_of_steps=[10 ** k for k in range(2, 6)],
        ),
        tasks.PointNucleusEnergyLevelsFindBoundState(
            name="PointNucleusEnergyLevelsFindBoundState (Task 3)",
            n_max=4,
            l_levels=range(0, 2 + 1),
            ngrid=20000,
            rmin=1e-15 * constants.A_BHOR,
        ),
        tasks.SmearedPotential(
            name="SmearedPotential (Task 4)", max_n_level=4, max_l_level=2, ngrid=40000
        ),
        tasks.Relativistic(
            name="Relativistic (Task 5)", max_n_level=4, max_l_level=2, ngrid=200000
        ),
    ]
    return [task for task in tasks_to_do if any(todo in task.name for todo in todos)]


def _print_hello_message():
    print(f"\n\t Numerov Solver for pi-12C system:")
    line = f"\n\t nucleus mass= {constants.N_NUCL} MeV \n\t pion    mass= {constants.M_PION} MeV"
    line = line + f"\n\t reduced mass= {constants.M_RED} MeV\n"
    print(line)


if __name__ == "__main__":
    main()

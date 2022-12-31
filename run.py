import tasks
import constants


def main():
    _print_hello_message()
    for task in _get_tasks():
        task.run("output")


def _get_tasks():
    return [
        tasks.PointNucleus(
            name="PointNucleus (Task 1)",
            rmin=1e-15 * constants.A_BHOR,
            rmax=10 * constants.A_BHOR,
            n_grid_points=int(1e3 + 1),
            energies=[-(0.9 + i * 0.05) * constants.RY for i in range(0, 5)],
        ),
        # tasks.Task2("Task 2"),
        # tasks.Task3("Task 3"),
        # tasks.Task4("Task 4"),
        # tasks.Task5("Task 5"),
    ]


def _print_hello_message():
    print(f"\n\t Numerov Solver for pi-12C system:")
    line = f"\n\t nucleus mass= {constants.N_NUCL} MeV \n\t pion    mass= {constants.M_PION} MeV"
    line = line + f"\n\t reduced mass= {constants.M_RED} MeV\n"
    print(line)


if __name__ == "__main__":
    main()

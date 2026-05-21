import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import get_robot_position

def convergence_graph(best_values, mean_values, worst_values, filename):
    os.makedirs("graficas", exist_ok=True)

    plt.figure(figsize=(10,6))

    plt.plot(best_values, label="Mejor fitness", linewidth=2)
    plt.plot(mean_values, label="Promedio fitness", linestyle="--")
    plt.plot(worst_values, label="Peor fitness", linestyle=":")

    plt.xlabel("Iteraciones")
    plt.ylabel("Valor de la función")
    plt.title("Evolución del Fitness - Algoritmo PSO")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"graficas/{filename}", dpi=300)
    plt.close()

def robot_graph(q, robot_links, target=None, filename="robot"):
    os.makedirs("graficas", exist_ok=True)

    points = get_robot_position(q, robot_links)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, marker="o", linewidth=2, label="Robot")

    if target is not None:
        ax.scatter(
            target[0],
            target[1],
            target[2],
            marker="x",
            s=100,
            label="Target"
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("Posición final del robot")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"graficas/{filename}", dpi=300)
    plt.close()
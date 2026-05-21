import numpy as np
import random
from utils import parse_robot, dh_matrix
from graficas import convergence_graph, robot_graph

cantidad_penalizaciones = 0
cantidad_movimientos = 0
suma_movimiento = 0

def fitness_robot(q, q_old, robot_links, target):
    current_position = forward_kinematics(q, robot_links, q_old)
    error = 1/(1+np.linalg.norm(abs(current_position-target)))
    return error

def forward_kinematics(q, robot_links, q_old = None):
    global cantidad_movimientos
    global suma_movimiento
    cantidad_movimientos += 1
    if q_old is not None:
        q_diff = q-q_old
        q_diff_mean = np.mean(q_diff)
        suma_movimiento += q_diff_mean

        if q_diff_mean > 0.000125:
            global cantidad_penalizaciones
            cantidad_penalizaciones += 1
            q = q+0.0005

    T = np.identity(4)
    for link in robot_links:
        if link["type"] == "R":
            A = dh_matrix(q[link["index"]], link["alpha"], link["b"], link["d"])
        else:
            A = dh_matrix(link["theta"], link["alpha"], link["b"], q[link["index"]])

        T = np.matmul(T, A)

    return T[:3, 3]

def pso(robot_links, target, bounds, num_particles=50, max_iter=75, w=0.75, c1=0.15, c2=0.15):
    random_individual = random.randint(0, num_particles)
    bounds = np.array(bounds)
    dimensions = len(bounds)
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    population = lower + np.random.rand(num_particles, dimensions) * (upper - lower)
    population_old = population.copy()
    velocity = np.random.uniform(-0.1, 0.1, (num_particles, dimensions))

    fitness = np.array([fitness_robot(population[i], population_old[i], robot_links, target) for i in range(len(population))])
    personal_best_position = population.copy()
    personal_best_fitness = fitness.copy()
    best_index = np.argmax(fitness)
    global_best_position = population[best_index].copy()
    global_best_fitness = fitness[best_index].copy()
    prev_best_fitness = 0

    best_values = []
    mean_values = []
    worst_values = []

    initial_best_individual = global_best_position.copy()
    random_individual_history = [np.float64(0)]

    j = 0

    while j < max_iter and global_best_fitness - prev_best_fitness > 0.00001:
        for i in range(num_particles):
            r1 = np.random.rand(dimensions)
            r2 = np.random.rand(dimensions)
            velocity[i] = (
                w * velocity[i]
                + c1 * r1 * (personal_best_position[i] - population[i])
                + c2 * r2 * (global_best_position - population[i])
            )

            population_old[i] = population[i].copy()
            population[i] = population[i] + velocity[i]
            population[i] = np.clip(population[i], lower, upper)
            fitness[i] = fitness_robot(population[i], population_old[i], robot_links, target)
            if fitness[i] > personal_best_fitness[i]:
                personal_best_fitness[i] = fitness[i]
                personal_best_position[i] = population[i].copy()
        best_index = np.argmax(personal_best_fitness)
        if fitness[best_index] > global_best_fitness:
            prev_best_fitness = global_best_fitness
            global_best_fitness = fitness[best_index].copy()
            global_best_position = population[best_index].copy()

        random_individual_history.append(np.mean(population[random_individual]) - np.mean(population_old[random_individual]))

        best_values.append(np.max(fitness))
        mean_values.append(np.mean(fitness))
        worst_values.append(np.min(fitness))

        j += 1

    print("Mejor fitness: ", global_best_fitness)
    print("Mejor q: ", global_best_position)
    print("Mejor Posicion: ", forward_kinematics(global_best_position, robot_links))
    print("Cantidad penalizaciones: ", cantidad_penalizaciones)
    print("Cantidad movimientos: ", cantidad_movimientos)
    print("Movimiento promedio: ", suma_movimiento/cantidad_movimientos)

    return {
        "best_values": best_values,
        "mean_values": mean_values,
        "worst_values": worst_values,
        "global_best_position": global_best_position,
        "initial_best_individual": initial_best_individual,
        "random_individual_history": random_individual_history,
    }

def main():
    robot_links, bounds, target = parse_robot("configuracion_robot.txt")

    result = pso(robot_links, target, bounds)
    #print(result)
    convergence_graph(result["best_values"], result["mean_values"], result["worst_values"], result["random_individual_history"], "convergence")
    robot_graph(result["global_best_position"], robot_links, target, "robot_final")
    robot_graph(result["initial_best_individual"], robot_links, target, "robot_initial")

main()
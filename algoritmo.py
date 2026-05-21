import numpy as np
import random
from utils import parse_robot, dh_matrix
from graficas import convergence_graph, robot_graph

# funcion fitness que regresa el error (se maximiza)
def fitness_robot(q, robot_links, target):
    current_position = forward_kinematics(q, robot_links)
    error = 1/(1+np.linalg.norm(abs(current_position-target)))
    return error

# funcion cinematica que hace la multiplicacion de matrices para calcular la
# posicion del ultimo componente del robot
def forward_kinematics(q, robot_links):
    # Matriz resultante, inicializada con matriz identidad
    T = np.identity(4)
    for link in robot_links:
        # Si la articulacion es rotacional se tiene que pasar q como theta
        if link["type"] == "R":
            A = dh_matrix(q[link["index"]], link["alpha"], link["b"], link["d"])
        # Si la articulacion es prismatica entonces se pasa q como d (displacement)
        else:
            A = dh_matrix(link["theta"], link["alpha"], link["b"], q[link["index"]])

        # Se multiplica y acumula el matriz que se acaba de sacar (A)
        T = np.matmul(T, A)

    return T[:3, 3]

# funcion metaheuristica que aproxima la cinematica inversa
def pso(robot_links, target, bounds, num_particles=50, max_iter=75, w=0.75, c1=0.15, c2=0.15):
    # Variables e inicializacion
    random_individual = random.randint(0, num_particles)
    bounds = np.array(bounds)
    dimensions = len(bounds)
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    population = lower + np.random.rand(num_particles, dimensions) * (upper - lower)
    velocity = np.random.uniform(-0.1, 0.1, (num_particles, dimensions))

    fitness = np.array([fitness_robot(population[i], robot_links, target) for i in range(len(population))])
    personal_best_position = population.copy()
    personal_best_fitness = fitness.copy()
    best_index = np.argmax(fitness)
    global_best_position = population[best_index].copy()
    global_best_fitness = fitness[best_index].copy()
    prev_best_fitness = 0

    # datos para graficas
    best_values = []
    mean_values = []
    worst_values = []

    initial_best_individual = global_best_position.copy()

    j = 0

    # ciclo principal que cierra con poco convergencia o un maximo de iteraciones
    while j < max_iter and global_best_fitness - prev_best_fitness > 0.00001:
        # ciclo for que corre num_particulas veces, 
        # pero no asegura que usa todas las particulas 1 vez
        for i in range(num_particles):
            # Seleccion de dos particulas random
            r1 = np.random.rand(dimensions)
            r2 = np.random.rand(dimensions)
            # Calcular velocidad (logica principal)
            velocity[i] = (
                w * velocity[i]
                + c1 * r1 * (personal_best_position[i] - population[i])
                + c2 * r2 * (global_best_position - population[i])
            )

            # actualizar posicion de la particula
            population[i] = population[i] + velocity[i]

            # bounds y actualizacion de fitness
            population[i] = np.clip(population[i], lower, upper)
            fitness[i] = fitness_robot(population[i], robot_links, target)
            if fitness[i] > personal_best_fitness[i]:
                personal_best_fitness[i] = fitness[i]
                personal_best_position[i] = population[i].copy()

        # actualizar el mejor individuo al final de la iteracion
        best_index = np.argmax(personal_best_fitness)
        if fitness[best_index] > global_best_fitness:
            prev_best_fitness = global_best_fitness
            global_best_fitness = fitness[best_index].copy()
            global_best_position = population[best_index].copy()

        # actualizar datos para graficas
        best_values.append(np.max(fitness))
        mean_values.append(np.mean(fitness))
        worst_values.append(np.min(fitness))

        j += 1

    # resultados
    print("Mejor fitness: ", global_best_fitness)
    print("Mejor q: ", global_best_position)
    print("Mejor Posicion: ", forward_kinematics(global_best_position, robot_links))

    return {
        "best_values": best_values,
        "mean_values": mean_values,
        "worst_values": worst_values,
        "global_best_position": global_best_position,
    }

def main():
    robot_links, bounds, target = parse_robot("configuracion_robot.txt")

    # ejecucion del algoritmo
    result = pso(robot_links, target, bounds)

    #graficas
    convergence_graph(result["best_values"], result["mean_values"], result["worst_values"], "convergence")
    robot_graph(result["global_best_position"], robot_links, target, "robot_final")

main()
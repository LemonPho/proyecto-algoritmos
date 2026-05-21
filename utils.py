import numpy as np

def dh_matrix(theta, alpha, b, d):
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), b * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), b * np.sin(theta)],
        [0,              np.sin(alpha),                 np.cos(alpha),                d],
        [0,              0,                             0,                            1]
    ])

def get_robot_position(q, robot_links):
    T = np.identity(4)
    points = []

    points.append(T[:3, 3].copy())

    for link in robot_links:
        if link["type"] == "R":
            A = dh_matrix(q[link["index"]], link["alpha"], link["b"], link["d"])
        else:
            A = dh_matrix(link["theta"], link["alpha"], link["b"], q[link["index"]])

        T = np.matmul(T, A)

        points.append(T[:3, 3].copy())

    return np.array(points)

def parse_robot(filename):
    robot_links = []
    bounds = []
    target = None

    variable_index = 0

    with open(filename, "r") as file:
        lines = file.readlines()

    reading_target = False

    for line in lines:
        line = line.strip()

        # ignorar líneas vacías y comentarios
        if not line or line.startswith("#"):
            continue

        # detectar sección TARGET
        if line.upper() == "TARGET":
            reading_target = True
            continue

        # leer target
        if reading_target:
            x, y, z = map(float, line.split())
            target = np.array([x, y, z])
            continue

        # leer parámetros D-H
        parts = line.split()

        theta, alpha, b, d, joint_type, min_val, max_val = parts

        alpha = float(alpha)
        b = float(b)

        min_val = float(min_val)
        max_val = float(max_val)

        link = {
            "type": joint_type,
            "alpha": alpha,
            "b": b,
            "index": variable_index
        }

        # articulación rotacional
        if joint_type == "R":
            link["d"] = float(d)

        # articulación prismática
        elif joint_type == "P":
            link["theta"] = float(theta)

        else:
            raise ValueError(f"Tipo de articulación inválido: {joint_type}")

        robot_links.append(link)
        bounds.append((min_val, max_val))

        variable_index += 1

    return robot_links, bounds, target
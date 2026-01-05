import random
import time

# --- CONFIGURACION DEL PROBLEMA ---
LIMITE_INFERIOR = -100.0
LIMITE_SUPERIOR = 100.0

def calcular_coste(solucion):
    """
    Funcion esfera: f(x) = sum(x_i^2)
    Minimo global: 0 en [0,0,...]
    """
    return sum(x**2 for x in solucion)

def generar_individuo(n):
    """Genera un individuo aleatorio de n dimensiones"""
    return [random.uniform(LIMITE_INFERIOR, LIMITE_SUPERIOR) for _ in range(n)]

def generar_poblacion(tamano_poblacion, n):
    """Genera la poblacion inicial"""
    return [generar_individuo(n) for _ in range(tamano_poblacion)]

def seleccion_torneo(poblacion, k=8):
    """Selecciona un individuo usando torneo"""
    torneo = random.sample(poblacion, k)
    torneo.sort(key=calcular_coste)
    return torneo[0]

def cruce_uniforme(padre1, padre2):
    """Cruza dos individuos usando cruce uniforme"""
    hijo = []
    for x, y in zip(padre1, padre2):
        hijo.append(x if random.random() < 0.5 else y)
    return hijo

def mutacion(individuo, prob_mutacion, sigma=0.5):
    """Aplica mutacion gaussiana a cada gen con cierta probabilidad"""
    hijo = []
    for x in individuo:
        if random.random() < prob_mutacion:
            x += random.gauss(0, sigma)
            # Limitar dentro de los rangos
            x = max(min(x, LIMITE_SUPERIOR), LIMITE_INFERIOR)
        hijo.append(x)
    return hijo

def algoritmo_genetico(n, tamano_poblacion=50, generaciones=200, sigma=0.5):
    # 1. Generar poblacion inicial
    poblacion = generar_poblacion(tamano_poblacion, n)

    mejor_individuo = min(poblacion, key=calcular_coste)
    mejor_fitness = calcular_coste(mejor_individuo)

    for gen in range(generaciones):
        nueva_poblacion = []

        while len(nueva_poblacion) < tamano_poblacion:
            # Seleccion de padres
            padre1 = seleccion_torneo(poblacion)
            padre2 = seleccion_torneo(poblacion)

            # Cruce
            hijo = cruce_uniforme(padre1, padre2)

            # Mutacion
            hijo = mutacion(hijo, 1/n, sigma)

            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

        # Actualizar mejor individuo
        actual_mejor = min(poblacion, key=calcular_coste)
        actual_fitness = calcular_coste(actual_mejor)
        if actual_fitness < mejor_fitness:
            mejor_individuo = actual_mejor
            mejor_fitness = actual_fitness

    return mejor_individuo, mejor_fitness

# --- EJECUCION ---
if __name__ == "__main__":
    initial_time = time.time()
    mejor_sol, mejor_valor = algoritmo_genetico(
        n=2,
        tamano_poblacion=50,
        generaciones=1000,
        sigma=0.5
    )
    end_time = time.time()

    print("\n--- RESULTADOS ALGORITMO GENETICO ---")
    print(f"Mejor fitness: {mejor_valor:.12f}")
    print(f"Mejor solucion: {[round(x,6) for x in mejor_sol]}")
    print(f"Tiempo de ejecucion: {end_time - initial_time:.6f} segundos")

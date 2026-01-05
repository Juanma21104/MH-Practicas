import math
import random
import time

# --- CONFIGURACION DEL PROBLEMA ---
LIMITE_INFERIOR = -100.0
LIMITE_SUPERIOR = 100.0
PASO_VECINO = 0.5  # Que tan lejos saltamos para buscar un vecino

def calcular_coste(solucion):
    """
    Funcion esfera: f(x) = sum(x_i^2)
    El objetivo es minimizar, el optimo es 0 en [0,0,...,0]
    """
    return sum(x**2 for x in solucion)

def generar_solucion_inicial(n):
    """Genera un vector de n dimensiones con valores aleatorios"""
    return [random.uniform(LIMITE_INFERIOR, LIMITE_SUPERIOR) for _ in range(n)]

def generar_vecino(solucion_actual):
    """
    Genera un vecino aplicando una pequeña perturbacion Gaussiana
    a cada dimension de la solucion actual
    """
    vecino = []
    for x in solucion_actual:
        # Añadimos ruido gaussiano (media 0, desviacion estandar PASO_VECINO)
        nuevo_valor = x + random.gauss(0, PASO_VECINO)
        
        # Aseguramos que no se salga de los limites
        if nuevo_valor < LIMITE_INFERIOR:
            nuevo_valor = LIMITE_INFERIOR
        elif nuevo_valor > LIMITE_SUPERIOR:
            nuevo_valor = LIMITE_SUPERIOR
            
        vecino.append(nuevo_valor)
    return vecino

# --- Recocido simulado ---
def recocido_simulado(n, t_inicial, alpha, max_iter=100000):
    # 1. Escoger solución inicial
    actual_x = generar_solucion_inicial(n)
    actual_f = calcular_coste(actual_x)
    
    mejor_x = list(actual_x)
    mejor_f = actual_f
    
    t_actual = t_inicial
    iteracion = 0
    
    # Bucle unico: hasta que se satisfaga criterio de parada
    while t_actual > 0.001 and mejor_f > 0 and iteracion < max_iter:
        
        # 2. Generar vecino
        vecino_x = generar_vecino(actual_x)
        vecino_f = calcular_coste(vecino_x)
        
        delta_f = vecino_f - actual_f
        
        aceptar = False
        
        # Criterio de aceptacion (metropolis)
        if delta_f <= 0:
            aceptar = True
        else:
            # Probabilidad p = exp(-delta / T)
            try:
                prob = math.exp(-delta_f / t_actual)
            except OverflowError:
                prob = 0
            if random.random() < prob:
                aceptar = True
        
        if aceptar:
            actual_x = vecino_x
            actual_f = vecino_f
            if actual_f < mejor_f:
                mejor_x = list(actual_x)
                mejor_f = actual_f
        
        # Actualizacion de temperatura (enfriamiento)
        t_actual = t_actual * alpha 
        iteracion += 1
        
    return mejor_x, mejor_f, iteracion

# --- EJECUCION ---
if __name__ == "__main__":
    # Parametros: 
    # n=2 dimensiones
    # Temperatura alta para permitir exploracion inicial
    # Alpha lento (0.99) para enfriar despacio y afinar la busqueda
    
    initial_time = time.time()
    mejor_solucion, mejor_fitness, iters = recocido_simulado(
        n=2, 
        t_inicial=100.0, 
        alpha=0.999, 
        max_iter=50000
    )
    end_time = time.time()

    print("\n--- RESULTADOS ---")
    print(f"Iteraciones: {iters}")
    print(f"Mejor fitness: {mejor_fitness:.10f}")
    print(f"Mejor solucion: {[round(x, 4) for x in mejor_solucion]}")
    print(f"Tiempo de ejecucion: {end_time - initial_time:.6f} segundos")
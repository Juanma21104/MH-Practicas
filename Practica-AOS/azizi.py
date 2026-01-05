import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# --- Función Objetivo (Cost Function) ---
def sphere(x):
    return np.sum(x**2)

# Asignación de la función de costo
cost_function = sphere

# --- Información del Problema ---
var_number = 2                      # Número de variables
var_min_val = -100
var_max_val = 100
var_min = var_min_val * np.ones(var_number)
var_max = var_max_val * np.ones(var_number)

# --- Parámetros Generales del Algoritmo ---
max_fes = 1000       # Máximo número de evaluaciones de función
n_pop = 50             # Tamaño de la población (candidatos)
layer_number = 5       # Máximo número de capas alrededor del núcleo
foton_rate = 0.1       # Tasa de fotones

# --- Contadores ---
iter_count = 0
fes = 0

# --- Inicialización ---
pop = np.zeros((n_pop, var_number))
cost = np.zeros(n_pop)

for i in range(n_pop):
    # Posiciones iniciales aleatorias
    pop[i, :] = np.random.uniform(var_min, var_max)
    # Evaluación de la función de costo
    cost[i] = cost_function(pop[i, :])
    fes += 1

# Ordenar Población
sort_order = np.argsort(cost)
cost = cost[sort_order]
pop = pop[sort_order, :]

best_pop = pop[0, :].copy()
mean_pop = np.mean(pop, axis=0)

# Historial para graficar convergencia
best_costs_history = []

print("Iniciando optimización...")

# --- Bucle Principal (Main Loop) ---
while fes < max_fes:
    iter_count += 1
    
    # Listas temporales para la nueva población
    pop_c_list = []
    cost_c_list = []
    
    # --- Crear Capas Cuánticas (Quantum Layers) ---
    # randi(LayerNumber) en Matlab es 1 a LayerNumber
    max_lay = np.random.randint(1, layer_number + 1) 
    
    nor_disp_input = np.arange(1, max_lay + 1)
    mu = 0
    sigma = max_lay / 6
    
    # Calcular PDF (Probability Density Function) similar a makedist/pdf en Matlab
    nor_disp = norm.pdf(nor_disp_input, loc=mu, scale=sigma)
    
    # Cálculos de normalización y distribución de partículas
    # Nota: Usamos listas o arrays temporales para simular los pasos de Matlab
    nor_disp_cal_2 = nor_disp / np.sum(nor_disp)
    nor_disp_cal_3 = n_pop * nor_disp_cal_2
    nor_disp_cal_4 = np.round(nor_disp_cal_3)
    nor_disp_cal_5 = np.cumsum(nor_disp_cal_4).astype(int)
    
    # Definir límites de las capas (indices)
    lay_col = np.concatenate(([0], nor_disp_cal_5))
    lay_col[lay_col > n_pop] = n_pop # Clamping
    
    # --- Bucle de Búsqueda (Search Loop) ---
    for i in range(max_lay):
        # Índices para slicing (Python es 0-based, Matlab 1-based, ajuste automático por lógica de slice)
        idx_start = lay_col[i]
        idx_end = lay_col[i+1]
        
        # Si la capa está vacía debido al redondeo, saltamos
        if idx_start >= idx_end:
            continue
            
        pop_a = pop[idx_start:idx_end, :]
        cost_a = cost[idx_start:idx_end]
        
        # Si la capa no está vacía
        energy = np.mean(cost_a)
        orbit = i + 1 # +1 porque en Python i empieza en 0
        
        # Arrays temporales para esta capa
        pop_b_layer = np.zeros_like(pop_a)
        cost_b_layer = np.zeros_like(cost_a)
        
        for j in range(len(pop_a)):
            if np.random.rand() > foton_rate:
                # Caso: Estado excitado o cambio de órbita
                ir = np.random.uniform(0, 1, 2)
                jr = np.random.uniform(0, 1, var_number)
                x_old = pop_a[j, :]
                
                if cost_a[j] > energy:
                    # Moverse basado en el mejor global y la media global
                    x_best = best_pop
                    x_mean = mean_pop
                    
                    delta = jr * (ir[0] * x_best - ir[1] * x_mean) / orbit
                    new_pos = x_old + delta
                else:
                    # Moverse basado en el mejor de la capa (PopA[0] ya que está ordenado al inicio)
                    x_best = pop_a[0, :]
                    
                    if len(pop_a) == 1:
                        x_mean = pop_a[0, :]
                    else:
                        x_mean = np.mean(pop_a, axis=0)
                        
                    delta = jr * (ir[0] * x_best - ir[1] * x_mean)
                    new_pos = x_old + delta
                
                # Aplicar límites (Boundary Check)
                new_pos = np.maximum(new_pos, var_min)
                new_pos = np.minimum(new_pos, var_max)
                
                pop_b_layer[j, :] = new_pos
                cost_b_layer[j] = cost_function(new_pos)
                fes += 1
                
            else:
                # Caso: Emisión/Absorción (Reinicio aleatorio)
                new_pos = np.random.uniform(var_min, var_max)
                pop_b_layer[j, :] = new_pos
                cost_b_layer[j] = cost_function(new_pos)
                fes += 1
        
        # Almacenar resultados de esta capa
        pop_c_list.append(pop_b_layer)
        cost_c_list.append(cost_b_layer)

    # --- Merge Candidates ---
    if pop_c_list: # Verificar si se generaron candidatos
        pop_c = np.vstack(pop_c_list)
        cost_c = np.concatenate(cost_c_list)
        
        pop = pop_c
        cost = cost_c
    
    # --- Sort Population ---
    sort_order = np.argsort(cost)
    pop = pop[sort_order, :]
    cost = cost[sort_order]
    
    # Mantener tamaño de población constante
    pop = pop[:n_pop, :]
    cost = cost[:n_pop]
    
    # Actualizar Mejor Global y Media
    best_pop = pop[0, :].copy()
    best_cost = cost[0]
    mean_pop = np.mean(pop, axis=0)
    
    # Guardar historial
    best_costs_history.append(best_cost)
    
    # Mostrar información de iteración
    print(f"Iteration {iter_count}: Best Cost = {best_cost:.10e}")

# --- Resultados Finales ---
eval_number = fes
conv_history = np.array(best_costs_history)
best_pos = best_pop

print("\nOptimización Finalizada.")
print(f"Mejor Costo Encontrado: {best_cost}")
print(f"Posición: {best_pos}")

# --- Gráfica de Convergencia (Opcional) ---
plt.figure()
plt.semilogy(conv_history)
plt.xlabel('Iteraciones')
plt.ylabel('Costo (Log Scale)')
plt.title('Convergencia AOS')
plt.grid(True)
plt.show()
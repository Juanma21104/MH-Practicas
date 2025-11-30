import numpy as np
import matplotlib.pyplot as plt
import time

class AtomSearchOptimizer:
    """
    Implementacion del algoritmo Atom Search Optimization (ASO) para problemas
    de optimizacion global
    
    El algoritmo simula el movimiento atomico basado en dinamicas moleculares,
    utilizando potenciales de interaccion (Lennard-Jones) y fuerzas de restriccion
    """
    
    def __init__(self, objective_function, dimension, n_atoms=50, max_iter=1000, 
                 lower_bound=-100, upper_bound=100, alpha=50, beta=0.2):
        """
        Inicializa el optimizador ASO

        Args:
            objective_function (callable): La funcion objetivo a minimizar
            dimension (int): Dimension del espacio de busqueda
            n_atoms (int): Tamaño de la poblacion de atomos
            max_iter (int): Numero maximo de iteraciones
            lower_bound (float/array): Limite inferior del espacio de busqueda
            upper_bound (float/array): Limite superior del espacio de busqueda
            alpha (float): Peso de profundidad para la fuerza de interaccion
            beta (float): Peso multiplicador para la fuerza de restriccion
        """
        self.obj_func = objective_function
        self.dim = dimension
        self.n_atoms = n_atoms
        self.max_iter = max_iter
        self.lb = lower_bound
        self.ub = upper_bound
        self.alpha = alpha
        self.beta = beta
        
        # Inicializacion del estado del sistema
        self.positions = np.random.uniform(self.lb, self.ub, (n_atoms, dimension))
        self.velocities = np.random.uniform(self.lb, self.ub, (n_atoms, dimension))
        
        # Almacenamiento de metricas
        self.fitness = np.zeros(n_atoms)
        self.masses = np.zeros(n_atoms)
        
        # Mejor solucion global encontrada
        self.global_best_pos = None
        self.global_best_fit = float('inf')
        
        # Historial para analisis y visualizacion
        self.history_positions = []
        self.history_fitness = []

    def calculate_potential_LJ(self, r, sigma, current_iter):
        """Calcula el potencial de interaccion basado en la distancia entre atomos."""
        # Funcion eta
        eta = self.alpha * (1 - (current_iter - 1) / self.max_iter) ** 3 * np.exp(-20 * current_iter / self.max_iter)
        
        # Rango dinamico de interaccion
        h_min = 1.1 + 0.1 * np.sin((current_iter / self.max_iter) * (np.pi / 2))
        h_max = 1.24
        
        # Ratio normalizado de distancia
        if sigma == 0:
            ratio = h_max
        else:
            ratio = r / sigma

        # Aplicar recorte (clipping) al ratio
        if ratio < h_min:
            h = h_min
        elif ratio > h_max:
            h = h_max
        else:
            h = ratio

        # Formula del potencial L-J modificada para optimizacion
        potential = eta * (2 * h**(13) - h**(7))
        return potential

    def calculate_masses(self):
        """Calcula la masa de cada atomo"""
        fit_best = self.global_best_fit
        fit_worst = np.max(self.fitness)
        
        if fit_worst == fit_best:
            self.masses = np.ones(self.n_atoms)
        else:
            self.masses = np.exp(-(self.fitness - fit_best) / (fit_worst - fit_best))
            self.masses = self.masses / np.sum(self.masses)

    def calculate_accelerations(self, current_iter):
        """Calcula las aceleraciones resultantes de las fuerzas de interaccion y restriccion."""
        self.calculate_masses()
        
        # Numero de vecinos K-best
        k_neighbors = int(self.n_atoms - (self.n_atoms - 2) * np.sqrt(current_iter / self.max_iter))
        
        # Ordenar atomos por fitness (mejores primero)
        sorted_indices = np.argsort(self.fitness)
        
        accelerations = np.zeros((self.n_atoms, self.dim))
        
        for i in range(self.n_atoms):
            interaction_force = np.zeros(self.dim)
            
            # Subconjunto de vecinos (K-best)
            neighbors_idx = sorted_indices[:k_neighbors]
            virtual_center = np.mean(self.positions[neighbors_idx], axis=0)
            
            # Escala de distancia sigma (distancia al centro de los vecinos)
            sigma = np.linalg.norm(self.positions[i] - virtual_center)
            
            # Sumatoria de fuerzas de interacción
            for neighbor_idx in neighbors_idx:
                if i == neighbor_idx: continue
                
                # Vector dirección y distancia
                diff_vector = self.positions[neighbor_idx] - self.positions[i]
                dist = np.linalg.norm(diff_vector)
                
                # Calculo del potencial y magnitud de fuerza
                potential = self.calculate_potential_LJ(dist, sigma, current_iter)
                
                # Perturbacion
                random_perturbation = np.random.rand(self.dim)
                
                # Acumular fuerza (evitando division por cero)                  Normalizamos
                force_component = random_perturbation * potential * (diff_vector / (dist + 1e-10))
                interaction_force += force_component
            
            # Fuerza total = fuerza interaccion + fuerza restriccion
            lambda_t = self.beta * np.exp(-20 * current_iter / self.max_iter)
            restriction_force = lambda_t * (self.global_best_pos - self.positions[i])
            total_force = interaction_force + restriction_force
            
            # Segunda ley de Newton: a = F / m
            if self.masses[i] > 1e-10:
                accelerations[i] = total_force / self.masses[i]
            else:
                accelerations[i] = 0.0
                
        return accelerations

    def optimize(self):
        """Ejecuta el bucle principal de optimizacion"""
        print(f"Iniciando optimizacion ASO ({self.n_atoms} atomos, {self.max_iter} iteraciones)...")
        # --- Bucle principal ---

        # Guardar historial inicial
        for i in range(self.n_atoms):
            self.fitness[i] = self.obj_func(self.positions[i])   
        best_idx = np.argmin(self.fitness)
        self.history_positions.append(self.positions.copy())
        self.history_fitness.append(self.fitness[best_idx])

        for t in range(0, self.max_iter):
            # Evaluacion de fitness
            for i in range(self.n_atoms):
                self.fitness[i] = self.obj_func(self.positions[i])
            
            # Actualizacion del mejor
            current_best_fit = np.min(self.fitness)
            current_best_idx = np.argmin(self.fitness)
            
            if current_best_fit < self.global_best_fit:
                self.global_best_fit = current_best_fit
                self.global_best_pos = self.positions[current_best_idx].copy()
            
            # Guardar historial
            self.history_positions.append(self.positions.copy())
            self.history_fitness.append(self.global_best_fit)
            
            # Calculo de aceleracion
            acc = self.calculate_accelerations(t)
            
            # Actualizacion de velocidad y posicion
            r_vec = np.random.rand(self.n_atoms, self.dim)
            self.velocities = r_vec * self.velocities + acc
            self.positions = self.positions + self.velocities
            
            # Control de fronteras (reinicializacion aleatoria)
            is_out = np.logical_or(self.positions < self.lb, self.positions > self.ub)
            random_positions = np.random.uniform(self.lb, self.ub, (self.n_atoms, self.dim))
            self.positions = np.where(is_out, random_positions, self.positions)
            
            # Log de progreso
            if t % 50 == 0 or t == self.max_iter:
                print(f"Iteracion {t}: Mejor Fitness = {self.global_best_fit:.6e}")
                
        return self.global_best_pos, self.global_best_fit


# ------------------------------------------------------------------------------
# DEFINICION DEL PROBLEMA Y EJECUCION
# ------------------------------------------------------------------------------

def sphere_function(x):
    """Funcion de prueba esfera. Optimo global f(0,..,0) = 0"""
    return np.sum(x**2)

# Configuracion del experimento
config = {
    'objective_function': sphere_function,
    'dimension': 2,
    'n_atoms': 50,
    'max_iter': 500,
    'lower_bound': -100,
    'upper_bound': 100,
    'alpha': 50,
    'beta': 0.2
}

# Instanciar y ejecutar
optimizer = AtomSearchOptimizer(**config)
initial_time = time.time()
best_solution, best_fitness = optimizer.optimize()
end_time = time.time()

print("\n" + "="*40)
print(f"RESULTADO FINAL")
print("="*40)
print(f"Posicion optima : {best_solution}")
print(f"Fitness optimo  : {best_fitness:.10e}")
print(f"Tiempo total    : {end_time - initial_time:.2f} segundos")


# ------------------------------------------------------------------------------
# VISUALIZACION DE RESULTADOS
# ------------------------------------------------------------------------------
def plot_results(optimizer_instance):

    """Genera dos figuras independientes: evolucion espacial y convergencia."""

   

    hist_pos = optimizer_instance.history_positions

    hist_fit = optimizer_instance.history_fitness

    lb, ub = optimizer_instance.lb, optimizer_instance.ub

   

    # Preparar malla para contorno de fondo

    x = np.linspace(lb, ub, 100)

    y = np.linspace(lb, ub, 100)

    X, Y = np.meshgrid(x, y)

    Z = X**2 + Y**2



    # --- FIGURA 1: Evolucion espacial de los atomos ---

    fig1 = plt.figure(figsize=(18, 6))

   

    snapshots = [0, len(hist_pos)//2, len(hist_pos)-1]

    titles = ["Fase Inicial (Exploracion)", "Fase Intermedia", "Fase Final (Explotacion)"]

   

    for i, iter_idx in enumerate(snapshots):

        ax = fig1.add_subplot(1, 3, i+1)

       

        # Fondo

        ax.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.4)

       

        # Atomos

        atoms = hist_pos[iter_idx]

        ax.scatter(atoms[:, 0], atoms[:, 1], c='crimson', s=25, alpha=0.8, label='Atomos')

       

        # Optimo teorico

        ax.scatter(0, 0, c='gold', marker='*', s=150, edgecolors='black', label='Optimo')

       

        ax.set_title(f"{titles[i]}\nIteracion {iter_idx}")

        ax.set_xlim(lb, ub)

        ax.set_ylim(lb, ub)

        ax.grid(True, linestyle='--', alpha=0.3)

       

        if i == 0:

            ax.legend(loc='upper right', framealpha=0.9)



    fig1.suptitle("Evolucion espacial del Algoritmo ASO", fontsize=16)

    plt.tight_layout()

    plt.show() # Mostrar primera figura

# Generar graficas
plot_results(optimizer)

# Esta funcion genera graficos con zoom automatico adaptado a la dispersion de los atomos
def plot_results_dynamic(optimizer_instance):
    hist_pos = optimizer_instance.history_positions
    hist_fit = optimizer_instance.history_fitness
    
    # Recuperamos los limites globales del problema
    global_lb = optimizer_instance.lb
    global_ub = optimizer_instance.ub
    
    fig1 = plt.figure(figsize=(18, 6))
    snapshots = [0, len(hist_pos)//2, len(hist_pos)-1]
    titles = ["Fase inicial (vista global)", "Fase intermedia", "Fase final"]
    
    for i, iter_idx in enumerate(snapshots):
        ax = fig1.add_subplot(1, 3, i+1)
        atoms = hist_pos[iter_idx]
        
        # --- LOGICA DE LIMITES ---
        if iter_idx == 0:
            # CASO 1: Inicio. Usamos los limites globales fijos
            xlims = [global_lb, global_ub]
            ylims = [global_lb, global_ub]
        else:
            # CASO 2: Resto. Calculamos zoom dinamico basado en los atomos
            min_x, max_x = atoms[:, 0].min(), atoms[:, 0].max()
            min_y, max_y = atoms[:, 1].min(), atoms[:, 1].max()
            
            # Asegurar que el optimo (0,0) salga en la foto
            min_x, max_x = min(min_x, 0), max(max_x, 0)
            min_y, max_y = min(min_y, 0), max(max_y, 0)
            
            # Padding
            range_x, range_y = max_x - min_x, max_y - min_y
            if range_x < 1e-15: range_x = 1e-5
            if range_y < 1e-15: range_y = 1e-5
            
            pad_x, pad_y = range_x * 0.2, range_y * 0.2
            
            xlims = [min_x - pad_x, max_x + pad_x]
            ylims = [min_y - pad_y, max_y + pad_y]

        # --- GENERACION DE FONDO ---
        # Creamos la malla segun los limites que hayamos decidido arriba
        x_grid = np.linspace(xlims[0], xlims[1], 100)
        y_grid = np.linspace(ylims[0], ylims[1], 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = X**2 + Y**2

        # Plot
        ax.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.4)
        ax.scatter(atoms[:, 0], atoms[:, 1], c='crimson', s=25, alpha=0.8, label='Atomos')
        ax.scatter(0, 0, c='gold', marker='*', s=150, edgecolors='black', label='Optimo')
        
        ax.set_title(f"{titles[i]}\nIteracion {iter_idx}")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        # Formato cientifico solo si hacemos zoom muy pequeño
        if iter_idx != 0: 
            ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 2))
            
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right')

    fig1.suptitle("Evolucion de ASO", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Convergencia
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(hist_fit, color='darkblue', linewidth=2)
    plt.title("Curva de convergencia", fontsize=14)
    plt.xlabel("Iteraciones"); plt.ylabel("Fitness")
    plt.yscale('log')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.show()

# Ejecutar
plot_results_dynamic(optimizer)
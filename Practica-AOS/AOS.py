import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

def _default_pdf(inp, layers):
    """
    Función PDF por defecto para distribuir las soluciones en orbitas
    """
    mu = 0
    sigma = layers / 6
    return norm.pdf(inp, loc=mu, scale=sigma)

class AtomicOrbitalSearch:
    """
    Implementacion del algoritmo Atomic Orbital Search (AOS) para problemas
    de optimizacion global.
    """
 
    def __init__(self, objective_function, lower_bounds, upper_bounds, n_electrons=50, 
                max_layers=5, max_iter=150000, photon_rate=0.9, pdf=_default_pdf):
        """
        Inicializa el optimizador AOS

        Args:
            objective_function (callable): La funcion objetivo a minimizar
            lower_bound (float/array): Limite inferior del espacio de busqueda (Tambien define la dimension del espacio de busqueda)
            upper_bound (float/array): Limite superior del espacio de busqueda (Tambien define la dimension del espacio de busqueda)
            n_electrons (int): Tamaño de la poblacion de electrones
            max_layers (int): Número maximo de órbitas
            max_iter (int): Numero maximo de iteraciones
            photon_rate (float): Probabilidad de actualización por interaccion con fotones
            pdf (callable): La funcion de densidad de probabilidad empleada para distribuir los electrones. 
                            Recibe como argumentos un array de valores a evaluar y el número de orbitas
        """
        if(len(lower_bounds) != len(upper_bounds)):
            raise Exception("Lower bound and upper bound dimension do not match!")

        self.obj_func = objective_function
        self.lb = lower_bounds
        self.ub = upper_bounds
        self.dim = len(lower_bounds)
        self.n_electrons = n_electrons
        self.max_layers = max_layers
        self.max_iter = max_iter
        self.pr = photon_rate
        self.pdf = pdf
        
        # Inicializacion del estado del sistema
        self.electrons = np.random.uniform(self.lb, self.ub, (n_electrons, self.dim))
        self.energy_levels = np.apply_along_axis(self.obj_func, 1, self.electrons)
        
        # Historial para analisis y visualizacion
        self.history_positions = []
        self.history_fitness = []

        # Determinar el Binding State (BS) y Lowest Energy (LE) del átomo 
        self._set_current_bs_and_le()

        self.print_state()

    def _set_current_bs_and_le(self):
        # Ordenar los electrones en base al nivel de energia
        sort_indexes = np.argsort(self.energy_levels)
        self.energy_levels = self.energy_levels[sort_indexes]
        self.electrons = self.electrons[sort_indexes]

        # Determinar el Binding State (BS) y Lowest Energy (LE) del átomo 
        self.le = self.electrons[0].copy()
        self.bs = np.mean(self.electrons, axis=0)

        # Actualizar historial
        self.history_positions.append(self.electrons.copy())
        self.history_fitness.append(self.energy_levels[0])

    def photon_emission(self, electron, layer):
        """
        Mutacion de una solucion por el efecto de emision de un foton
        """
        alpha = np.random.uniform(size=self.dim)
        beta = np.random.uniform(size=self.dim)
        gamma = np.random.uniform(size=self.dim)

        p = alpha * (beta * self.le - gamma * self.bs)

        return electron + p / layer

    def photon_absorption(self, electron, le, bs):
        """
        Mutacion de una solucion por el efecto de absorcion de un foton
        """
        alpha = np.random.uniform(size=self.dim)
        beta = np.random.uniform(size=self.dim)
        gamma = np.random.uniform(size=self.dim)

        p = alpha * (beta * le - gamma * bs)

        return electron + p

    def magnetic_perturbation(self, electron):
        """
        Mutacion de una solucion por el efecto perturbacion aleatoria
        """
        return electron + np.random.uniform(size=self.dim)

    def print_state(self):
        """
        Impresion por salida estandar del estado del algoritmo
        """
        print("\n" + "="*40)
        print(self.electrons)
        print(self.energy_levels)
        print("LE: {}".format(self.le))
        print("BS: {}".format(self.bs))
        print("="*40)

    def optimize(self):
        """
        Ejecuta el bucle principal de optimizacion
        """
        print(f"Iniciando optimizacion AOS ({self.n_electrons} electrones, {self.max_iter} iteraciones)...")
        # --- Bucle principal ---
        for i in range(self.max_iter):
            #Generar orbitas
            n = np.random.randint(self.max_layers) + 1

            #Distribuir electrones en orbitas
            pdf_input = np.arange(1, n + 1)
            distribution = self.pdf(pdf_input, n)
            probabilities = distribution / np.sum(distribution)
            counts = np.round(probabilities * self.n_electrons)
  
            indexes = np.cumsum(counts)
            indexes = np.insert(indexes, 0, 0)
            indexes[indexes > self.n_electrons] = self.n_electrons

            layers = indexes.astype(int)

            for layer in range(1, n + 1):
                if layers[layer-1] >= self.n_electrons:
                    break

                bs = np.mean(self.electrons[layers[layer-1]:layers[layer]], axis=0)
                be = np.mean(self.energy_levels[layers[layer-1]:layers[layer]])
                le = self.electrons[layers[layer-1]].copy()

                for electron_index in range(layers[layer-1], layers[layer]):
                    electron = self.electrons[electron_index]
                    phi = np.random.rand()

                    if phi <= self.pr:
                        if self.energy_levels[electron_index] >= be:
                            #Emision foton
                            self.electrons[electron_index] = self.photon_emission(electron, layer)
                        else:
                            #Absorcion foton
                            self.electrons[electron_index] = self.photon_absorption(electron, le, bs)
                    else:
                        #Perturbacion aleatoria
                        self.electrons[electron_index] = self.magnetic_perturbation(electron)

                    #Manejo de los limites de las variables de decision
                    self.electrons[electron_index] = np.maximum(self.electrons[electron_index], self.lb)
                    self.electrons[electron_index] = np.minimum(self.electrons[electron_index], self.ub)

                    #Actualizacion del nivel de energia de la solucion
                    self.energy_levels[electron_index] = self.obj_func(self.electrons[electron_index])
                
            #Actualizar BS y LE
            self._set_current_bs_and_le()
        
        return self.le, self.obj_func(self.le)

# ------------------------------------------------------------------------------
# DEFINICION DEL PROBLEMA Y EJECUCION
# ------------------------------------------------------------------------------

def sphere_function(x):
    """Funcion de prueba esfera. Optimo global f(0,..,0) = 0"""
    return np.sum(x**2)

# Configuracion del experimento

lb = -100
ub = 100
config = {
    'objective_function': sphere_function,
    'lower_bounds': np.ones(2) * lb,
    'upper_bounds': np.ones(2) * ub,
    'n_electrons': 50,
    'max_layers': 5,
    'max_iter': 500,
    'photon_rate': 0.9
}

# Instanciar y ejecutar
optimizer = AtomicOrbitalSearch(**config)
initial_time = time.time()
best_solution, best_fitness = optimizer.optimize()
end_time = time.time()

optimizer.print_state()
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

    # Preparar malla para contorno de fondo
    lb = optimizer_instance.lb[0]
    ub = optimizer_instance.ub[0]

    x = np.linspace(lb, ub, 100)
    y = np.linspace(lb, ub, 100)

    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # --- FIGURA 1: Evolucion espacial de los electrones ---
    fig1 = plt.figure(figsize=(18, 6))   

    snapshots = [0, len(hist_pos)//2, len(hist_pos)-1]

    titles = ["Fase Inicial", "Fase Intermedia", "Fase Final"]

    for i, iter_idx in enumerate(snapshots):
        ax = fig1.add_subplot(1, 3, i+1)

        # Fondo
        ax.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.4)

        # Electrones
        electrons = hist_pos[iter_idx]
        ax.scatter(electrons[:, 0], electrons[:, 1], c='crimson', s=25, alpha=0.8, label='Electrones')

        # Optimo teorico
        ax.scatter(0, 0, c='gold', marker='*', s=150, edgecolors='black', label='Optimo')
        ax.set_title(f"{titles[i]}\nIteracion {iter_idx}")
        ax.set_xlim(lb, ub)
        ax.set_ylim(lb, ub)
        ax.grid(True, linestyle='--', alpha=0.3)     

        if i == 0:
            ax.legend(loc='upper right', framealpha=0.9)

    fig1.suptitle("Evolucion espacial del Algoritmo AOS", fontsize=16)

    plt.tight_layout()

    plt.show() # Mostrar primera figura

# Generar graficas
plot_results(optimizer)

# Esta funcion genera graficos con zoom automatico adaptado a la dispersion de los electrones
def plot_results_dynamic(optimizer_instance):
    hist_pos = optimizer_instance.history_positions
    hist_fit = optimizer_instance.history_fitness
    
    # Recuperamos los limites globales del problema
    global_lb = optimizer_instance.lb[0]
    global_ub = optimizer_instance.ub[0]
    
    fig1 = plt.figure(figsize=(18, 6))
    snapshots = [0, len(hist_pos)//2, len(hist_pos)-1]
    titles = ["Fase inicial (vista global)", "Fase intermedia", "Fase final"]
    
    for i, iter_idx in enumerate(snapshots):
        ax = fig1.add_subplot(1, 3, i+1)
        electrons = hist_pos[iter_idx]
        
        # --- LOGICA DE LIMITES ---
        if iter_idx == 0:
            # CASO 1: Inicio. Usamos los limites globales fijos
            xlims = [global_lb, global_ub]
            ylims = [global_lb, global_ub]
        else:
            # CASO 2: Resto. Calculamos zoom dinamico basado en los electrones
            min_x, max_x = electrons[:, 0].min(), electrons[:, 0].max()
            min_y, max_y = electrons[:, 1].min(), electrons[:, 1].max()
            
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
        ax.scatter(electrons[:, 0], electrons[:, 1], c='crimson', s=25, alpha=0.8, label='Electrones')
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

    fig1.suptitle("Evolucion de AOS", fontsize=16)
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
import numpy as np
import random

# Parâmetros do Problema e do PSO
N_QUEENS = 8
NUM_PARTICLES = 50
MAX_ITER = 1000

# Parâmetros que você deve "Variar" conforme a atividade pede
W = 1.1       # Peso de inércia
C1 = 1.5      # Constante cognitiva
C2 = 2.0    # Constante social

def calculate_conflicts(position):
    """Calcula o número de ataques (colisões) entre as rainhas."""
    # Arredonda e limita para garantir que estão dentro do tabuleiro [0, N-1]
    pos = np.clip(np.round(position), 0, N_QUEENS - 1).astype(int)
    conflicts = 0
    
    for i in range(N_QUEENS):
        for j in range(i + 1, N_QUEENS):
            # Verifica se estão na mesma linha
            if pos[i] == pos[j]:
                conflicts += 1
            # Verifica se estão na mesma diagonal
            elif abs(pos[i] - pos[j]) == abs(i - j):
                conflicts += 1
                
    return conflicts

class Particle:
    def __init__(self):
        # Inicia com posições aleatórias (0 a 7) para as 8 colunas
        self.position = np.random.uniform(0, N_QUEENS - 1, N_QUEENS)
        self.velocity = np.random.uniform(-1, 1, N_QUEENS)
        
        self.pbest_position = self.position.copy()
        self.pbest_fitness = calculate_conflicts(self.position)

def run_pso():
    particles = [Particle() for _ in range(NUM_PARTICLES)]
    
    # Encontrar o Global Best (gbest) inicial
    gbest_position = particles[0].position.copy()
    gbest_fitness = particles[0].pbest_fitness
    
    for p in particles:
        if p.pbest_fitness < gbest_fitness:
            gbest_fitness = p.pbest_fitness
            gbest_position = p.position.copy()
            
    # Loop de otimização
    iterations_done = 0
    for iteration in range(MAX_ITER):
        iterations_done = iteration
        # Se achou a solução ideal (0 colisões), pode parar
        if gbest_fitness == 0:
            print(f"Solução encontrada na iteração {iteration}!")
            break
            
        for p in particles:
            # 1. Atualizar velocidade
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_term = C1 * r1 * (p.pbest_position - p.position)
            social_term = C2 * r2 * (gbest_position - p.position)
            
            p.velocity = (W * p.velocity) + cognitive_term + social_term
            
            # Limitar a velocidade para não "explodir"
            p.velocity = np.clip(p.velocity, -N_QUEENS/2, N_QUEENS/2)
            
            # 2. Atualizar posição
            p.position = p.position + p.velocity
            
            # 3. Avaliar novo fitness
            current_fitness = calculate_conflicts(p.position)
            
            # 4. Atualizar Pbest
            if current_fitness < p.pbest_fitness:
                p.pbest_fitness = current_fitness
                p.pbest_position = p.position.copy()
                
                # 5. Atualizar Gbest
                if current_fitness < gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_position = p.position.copy()

    return np.clip(np.round(gbest_position), 0, N_QUEENS - 1).astype(int), gbest_fitness, iterations_done

# Executar o algoritmo 10 vezes
NUM_RUNS = 10
fitness_results = []
best_overall_solution = None
best_overall_fitness = float('inf')

for run in range(NUM_RUNS):
    best_solution, best_fitness, num_attempts = run_pso()
    fitness_results.append(best_fitness)
    
    # Manter a melhor solução encontrada entre todas as execuções
    if best_fitness < best_overall_fitness:
        best_overall_fitness = best_fitness
        best_overall_solution = best_solution

# Calcular a média dos ataques
average_fitness = np.mean(fitness_results)

print("\n--- Resultado Final ---")
print(f"Número de Execuções: {NUM_RUNS}")
print(f"Conflitos por Execução: {fitness_results}")
print(f"Média de Conflitos: {average_fitness:.2f}")
print(f"Melhor Resultado: {best_overall_fitness}")
print(f"Melhor Vetor de Posição (Linhas): {best_overall_solution}")

# Desenho visual do tabuleiro no console
if best_overall_fitness == 0:
    print("\nTabuleiro:")
    for row in range(N_QUEENS):
        line = ""
        for col in range(N_QUEENS):
            if best_overall_solution[col] == row:
                line += "[Q] "
            else:
                line += "[ ] "
        print(line)
else:
    print("\nO algoritmo ficou preso em um ótimo local (comum em metaheurísticas). Rode novamente ou ajuste os parâmetros.")
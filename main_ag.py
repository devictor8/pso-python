import random
import numpy as np

N_QUEENS = 8
MAX_GENERATIONS = 1000
NUM_RUNS = 10
CROSSOVER_RATE = 0.9
ELITISM = 1


def calculate_conflicts(individual):
    conflicts = 0

    for i in range(N_QUEENS):
        for j in range(i + 1, N_QUEENS):
            if individual[i] == individual[j]:
                conflicts += 1
            elif abs(individual[i] - individual[j]) == abs(i - j):
                conflicts += 1

    return conflicts


def create_individual():
    individual = list(range(N_QUEENS))
    random.shuffle(individual)
    return individual


def tournament_selection(population, tournament_size=3):
    selected = random.sample(population, tournament_size)
    return min(selected, key=calculate_conflicts).copy()


def order_crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    start, end = sorted(random.sample(range(N_QUEENS), 2))

    def create_child(p1, p2):
        child = [None] * N_QUEENS

        child[start:end + 1] = p1[start:end + 1]

        remaining_genes = [gene for gene in p2 if gene not in child]

        index = 0
        for i in range(N_QUEENS):
            if child[i] is None:
                child[i] = remaining_genes[index]
                index += 1

        return child

    child1 = create_child(parent1, parent2)
    child2 = create_child(parent2, parent1)

    return child1, child2


def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(N_QUEENS), 2)
        individual[i], individual[j] = individual[j], individual[i]

    return individual


def run_ga(population_size, mutation_rate):
    population = [create_individual() for _ in range(population_size)]

    best_individual = min(population, key=calculate_conflicts).copy()
    best_fitness = calculate_conflicts(best_individual)

    for generation in range(MAX_GENERATIONS):
        if best_fitness == 0:
            break

        new_population = sorted(population, key=calculate_conflicts)[:ELITISM]

        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            child1, child2 = order_crossover(parent1, parent2)

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.append(child1)

            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

        current_best = min(population, key=calculate_conflicts).copy()
        current_fitness = calculate_conflicts(current_best)

        if current_fitness < best_fitness:
            best_individual = current_best
            best_fitness = current_fitness

    return best_individual, best_fitness


def run_experiment(population_size, mutation_rate):
    fitness_results = []
    best_overall_solution = None
    best_overall_fitness = float("inf")

    for run in range(NUM_RUNS):
        best_solution, best_fitness = run_ga(population_size, mutation_rate)
        fitness_results.append(best_fitness)

        if best_fitness < best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_solution = best_solution

    average_fitness = np.mean(fitness_results)

    print("\n--------------------------------")
    print(f"População: {population_size}")
    print(f"Taxa de mutação: {mutation_rate * 100:.0f}%")
    print(f"Número de execuções: {NUM_RUNS}")
    print(f"Conflitos por execução: {fitness_results}")
    print(f"Média de conflitos: {average_fitness:.2f}")
    print(f"Melhor resultado: {best_overall_fitness}")
    print(f"Melhor vetor de posição: {best_overall_solution}")

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

    return average_fitness


configs = [
    (10, 0.20),
    (10, 0.05),
    (100, 0.05),
    (100, 0.20),
]

for population_size, mutation_rate in configs:
    run_experiment(population_size, mutation_rate)
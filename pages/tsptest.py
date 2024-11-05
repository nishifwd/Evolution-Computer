import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from itertools import permutations

# Setting a fixed number of cities to 9
num_cities = 9

# Collecting city names and coordinates
cities_names = []
x = []
y = []

# Create input fields for city names and coordinates
st.write("Enter the details for each city:")
columns = st.columns(3)
for i in range(num_cities):
    with columns[0]:
        cities_name = st.text_input(f"City {i+1} Name:", value=f"City{i+1}")
        cities_names.append(cities_name)
    with columns[1]:
        city_x = st.number_input(f"City {i+1} X-coordinate:", value=int(i * 2), key=f"x{i}")
        x.append(city_x)
    with columns[2]:
        city_y = st.number_input(f"City {i+1} Y-coordinate:", value=int(i), key=f"y{i}")
        y.append(city_y)

# Display the "Submit" button only after input is complete
if all(cities_names) and all(x) and all(y):
    if st.button("Submit"):
        # Constructing city coordinates dictionary
        city_coords = dict(zip(cities_names, zip(x, y)))

        # GA parameters
        n_population = 250
        crossover_per = 0.8
        mutation_per = 0.2
        n_generations = 200

        # Visualization colors and icons
        colors = sns.color_palette("pastel", len(cities_names))
        city_icons = {city: f"♕" for city in cities_names}  # Simple placeholder icon for visualization

        # Plotting the cities
        fig, ax = plt.subplots()
        ax.grid(False)

        for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
            color = colors[i % len(colors)]
            icon = city_icons[city]
            ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
            ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
            ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                        textcoords='offset points')

            for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
                if i != j:
                    ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

        fig.set_size_inches(16, 12)
        st.pyplot(fig)

        # GA functions and logic go here
        def initial_population(cities_list, n_population=250):
            population_perms = []
            possible_perms = list(permutations(cities_list))
            random_ids = random.sample(range(0, len(possible_perms)), n_population)
            for i in random_ids:
                population_perms.append(list(possible_perms[i]))
            return population_perms

        def dist_two_cities(city_1, city_2):
            city_1_coords = city_coords[city_1]
            city_2_coords = city_coords[city_2]
            return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

        def total_dist_individual(individual):
            total_dist = 0
            for i in range(len(individual)):
                if i == len(individual) - 1:
                    total_dist += dist_two_cities(individual[i], individual[0])
                else:
                    total_dist += dist_two_cities(individual[i], individual[i + 1])
            return total_dist

        def fitness_prob(population):
            total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
            max_population_cost = max(total_dist_all_individuals)
            population_fitness = max_population_cost - np.array(total_dist_all_individuals)
            population_fitness_sum = sum(population_fitness)
            return population_fitness / population_fitness_sum

        def roulette_wheel(population, fitness_probs):
            population_fitness_probs_cumsum = fitness_probs.cumsum()
            bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)
            selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1
            return population[selected_individual_index]

        def crossover(parent_1, parent_2):
            n_cities_cut = len(cities_names) - 1
            cut = round(random.uniform(1, n_cities_cut))
            offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
            offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
            return offspring_1, offspring_2

        def mutation(offspring):
            n_cities_cut = len(cities_names) - 1
            index_1, index_2 = random.sample(range(0, n_cities_cut + 1), 2)
            offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
            return offspring

        def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
            population = initial_population(cities_names, n_population)
            for i in range(n_generations):
                fitness_probs = fitness_prob(population)
                parents_list = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
                offspring_list = []
                for i in range(0, len(parents_list), 2):
                    if i + 1 < len(parents_list):
                        offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i + 1])
                        if random.random() > (1 - mutation_per):
                            offspring_1 = mutation(offspring_1)
                        if random.random() > (1 - mutation_per):
                            offspring_2 = mutation(offspring_2)
                        offspring_list.extend([offspring_1, offspring_2])

                mixed_offspring = parents_list + offspring_list
                fitness_probs = fitness_prob(mixed_offspring)
                sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
                best_mixed_offspring = [mixed_offspring[i] for i in sorted_fitness_indices[:n_population]]

            return best_mixed_offspring

        best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

        total_dist_all_individuals = [total_dist_individual(ind) for ind in best_mixed_offspring]
        index_minimum = np.argmin(total_dist_all_individuals)
        minimum_distance = min(total_dist_all_individuals)
        st.write("Minimum Distance: ", minimum_distance)

        # Shortest path
        shortest_path = best_mixed_offspring[index_minimum]
        st.write("Shortest Path: ", shortest_path)

        x_shortest = [city_coords[city][0] for city in shortest_path]
        y_shortest = [city_coords[city][1] for city in shortest_path]
        x_shortest.append(x_shortest[0])
        y_shortest.append(y_shortest[0])

        fig, ax = plt.subplots()
        ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
        plt.legend()

        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

        plt.title(label="TSP Best Route Using GA", fontsize=25, color="k")
        str_params = f'\n{n_generations} Generations\n{n_population} Population Size\n{crossover_per} Crossover\n{mutation_per} Mutation'
        plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}{str_params}", fontsize=18, y=1.047)

        for i, txt in enumerate(shortest_path):
            ax.annotate(f"{i + 1}- {txt}", (x_shortest[i], y_shortest[i]), fontsize=20)

        fig.set_size_inches(16, 12)
        st.pyplot(fig)

else:
    st.write("Please complete all fields before submitting.")

pip install pgmpy networkx matplotlib fpdf

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from networkx.algorithms import community as nx_community
from pgmpy.inference.EliminationOrder import MinFill
import random
import time


def prune_basic(model, query_node, evidence_nodes):
    relevant_nodes = set([query_node]) | set(evidence_nodes)
    for node in relevant_nodes.copy():
        relevant_nodes.update(nx.ancestors(nx.DiGraph(model.edges()), node))
    irrelevant_nodes = set(model.nodes()) - relevant_nodes
    pruned_model = model.copy()
    pruned_model.remove_nodes_from(irrelevant_nodes)
    return pruned_model, irrelevant_nodes


def prune_refined(model, query_node, evidence_nodes):
    pruned_model = model.copy()
    irrelevant_nodes = set()
    ancestors_query = nx.ancestors(nx.DiGraph(model.edges()), query_node)
    ancestors_evidence = set()
    for evidence_node in evidence_nodes:
        ancestors_evidence.update(nx.ancestors(nx.DiGraph(model.edges()), evidence_node))

    for node in model.nodes():
        if node != query_node and node not in evidence_nodes:
            node_ancestors = nx.ancestors(nx.DiGraph(model.edges()), node)
            if node_ancestors.isdisjoint({query_node}) and node_ancestors.isdisjoint(evidence_nodes):
                pruned_model.remove_node(node)
                irrelevant_nodes.add(node)
    return pruned_model, irrelevant_nodes


def prune_v_structure(model, query_var, evidence):
    pruned_model = model.copy()
    evidence_nodes = set(evidence.keys())  # Ensure evidence nodes are kept
    nodes_to_keep = set([query_var] + list(evidence_nodes))

    # Create a static copy for safe iteration
    static_nodes_to_keep = list(nodes_to_keep)

    # Include all ancestors of the query and evidence nodes in the nodes to keep
    for node in static_nodes_to_keep:
        ancestors = nx.ancestors(nx.DiGraph(model.edges()), node)
        nodes_to_keep.update(ancestors)

    irrelevant_nodes = set()

    # Check each node for being a part of a v-structure
    for node in list(pruned_model.nodes()):  # Use a static list to iterate safely
        if node in nodes_to_keep:
            continue  # Skip the critical nodes from any removal
        parents = model.get_parents(node)
        if len(parents) > 1:  # Node has more than one parent, potential v-structure center
            children = model.get_children(node)
            # Ensure no evidence nodes are removed as descendants
            if all(child not in nodes_to_keep for child in children):
                descendants = nx.descendants(nx.DiGraph(model.edges()), node)
                # Avoid removing any evidence nodes or the query node
                descendants = descendants - nodes_to_keep
                irrelevant_nodes.update(descendants)
                irrelevant_nodes.add(node)

                # Safely remove the center node and its descendants if they are not evidence nodes
                if node in pruned_model.nodes():
                    pruned_model.remove_node(node)  # Remove the center node if present
                # Safely remove descendants that still exist in the pruned model
                pruned_model.remove_nodes_from([desc for desc in descendants if desc in pruned_model.nodes()])

    return pruned_model, irrelevant_nodes


def prune_separators(model, query_var, evidence):
    pruned_model = model.copy()
    nx_graph = nx.DiGraph(model.edges())
    separators = set()  # Use a set to avoid duplicates

    # Identifying separators based on disconnection of the query from evidence
    for node in list(nx_graph.nodes()):
        if node != query_var and node not in evidence:
            temp_graph = nx_graph.copy()
            temp_graph.remove_node(node)
            # Check if removing the node disconnects any evidence node from the query
            if not all(nx.has_path(temp_graph, query_var, ev) for ev in evidence):
                separators.add(node)

    # Remove separators and their descendants, ensuring each node is only processed once
    for sep in list(separators):
        if sep in pruned_model.nodes():
            descendants = nx.descendants(nx_graph, sep)
            for desc in list(descendants):
                if desc in pruned_model.nodes() and desc != query_var:
                    pruned_model.remove_node(desc)
            if sep in pruned_model.nodes() and sep != query_var:
                pruned_model.remove_node(sep)

    return pruned_model, separators


def prune_clusters(model, query_var, evidence):
    pruned_model = model.copy()
    nx_graph = nx.Graph(model.edges())  # Convert to undirected graph for community detection

    # Using the asyn_lpa_communities algorithm from networkx for demonstration
    communities = list(nx_community.asyn_lpa_communities(nx_graph))
    query_community = None

    # Find the community that contains the query variable
    for community in communities:
        if query_var in community:
            query_community = community
            break

    # Retain only nodes within the same community as the query variable
    nodes_to_keep = set(query_community) if query_community else set()
    nodes_to_remove = set(model.nodes()) - nodes_to_keep

    for node in nodes_to_remove:
        if node != query_var:  # Ensure the query variable is not removed
            pruned_model.remove_node(node)

    return pruned_model, nodes_to_keep


def query_pruned_network(pruned_model, query_var, full_evidence):
    # Ensure the query variable is not removed
    if query_var not in pruned_model.nodes():
        raise ValueError(f"The query variable {query_var} was erroneously pruned.")

    # Filter the evidence to only include nodes that exist in the pruned model
    pruned_evidence = {node: val for node, val in full_evidence.items() if node in pruned_model.nodes()}

    pruned_inference = VariableElimination(pruned_model)
    result_pruned = pruned_inference.query(variables=[query_var], evidence=pruned_evidence)
    return result_pruned


def create_balanced_bayesian_network(num_nodes, edge_prob, max_parents=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    nodes = [f'X{i}' for i in range(num_nodes)]
    edges = []
    for i in range(1, num_nodes):
        possible_parents = nodes[:i]
        num_parents = min(len(possible_parents), max_parents)
        parents = random.sample(possible_parents, k=random.randint(1, num_parents))
        for parent in parents:
            edges.append((parent, nodes[i]))
    model = BayesianNetwork(edges)
    for node in model.nodes():
        num_parents = len(model.get_parents(node))
        # cpd_values = np.full((2, 2**num_parents), 1/2) #Uniform
        cpd_values = np.random.rand(2, 2**num_parents) #Random
        # cpd_values = np.abs(np.random.randn(2, 2**num_parents)) #Gaussian
        #cpd_values = np.array([np.random.dirichlet(np.ones(2**num_parents)) for _ in range(2)]) #Dirichlet
        cpd_values /= cpd_values.sum(axis=0)
        evidence = model.get_parents(node)
        evidence_card = [2] * len(evidence)
        cpd = TabularCPD(variable=node, variable_card=2, values=cpd_values, evidence=evidence, evidence_card=evidence_card)
        model.add_cpds(cpd)
    assert model.check_model(), "Model has errors"
    return model


def visualize_network(model, title):
    nx_graph = nx.DiGraph(model.edges())
    pos = nx.spring_layout(nx_graph)
    plt.figure(figsize=(8, 8))
    nx.draw(nx_graph, pos, with_labels=True, node_size=1000, node_color='lightblue', edge_color='k', font_size=12)
    plt.title(title)
    plt.show()


def calculate_cost(model, elimination_order):
    """ Calculate the cumulative cost of eliminating nodes based on the elimination order. """
    cost_calculator = MinFill(model)  # Create a MinFill object
    total_cost = sum(cost_calculator.cost(node) for node in elimination_order)
    return total_cost

def run_experiment(node_sizes, num_runs, num_repeats, evidence_no, edge_den, max_parents, seed=42):
    results = {}

    for size in node_sizes:
        print(f"\nRunning experiments for network size: {size}")
        costs = {'full': [], 'basic': [], 'refined': [], 'v_structure': [], 'separators': [], 'clusters': []}
        accuracies = {'basic': [], 'refined': [], 'v_structure': [], 'separators': [], 'clusters': []}
        nodes_pruned = {'basic': [], 'refined': [], 'v_structure': [], 'separators': [], 'clusters': []}
        pruning_times = {'basic': [], 'refined': [], 'v_structure': [], 'separators': [], 'clusters': []}

        for _ in range(num_runs):
            model = create_balanced_bayesian_network(size, edge_den, max_parents, seed)
            nodes = list(model.nodes())

            for _ in range(num_repeats):
                query_node = random.choice(nodes)
                potential_evidence_nodes = [node for node in nodes if node != query_node]
                evidence_nodes = random.sample(potential_evidence_nodes, k=int(len(nodes) * evidence_no))
                evidence = {node: np.random.randint(2) for node in evidence_nodes}

                # Full network inference and cost
                inference_full = VariableElimination(model)
                result_full = inference_full.query(variables=[query_node], evidence=evidence)
                elimination_order_full = MinFill(model).get_elimination_order()
                full_cost = calculate_cost(model, elimination_order_full)
                costs['full'].append(full_cost)

                # Pruning and inference for each method
                for method in ['basic', 'refined', 'v_structure', 'separators', 'clusters']:
                    start_prune = time.time()
                    pruned_model, pruned_nodes = globals()[f'prune_{method}'](model, query_node, evidence)
                    prune_time = time.time() - start_prune

                    try:
                        result_pruned = query_pruned_network(pruned_model, query_node, evidence)
                        pruned_cost = calculate_cost(pruned_model, MinFill(pruned_model).get_elimination_order())
                        accuracies[method].append(np.isclose(result_full.values, result_pruned.values, atol=1e-3).all())
                    except ValueError as e:
                        print(e)
                        continue  # Skip if the query variable was pruned in error

                    costs[method].append(pruned_cost)
                    nodes_pruned[method].append(len(pruned_nodes))
                    pruning_times[method].append(prune_time)

        results[size] = {
            'costs': costs,
            'accuracies': accuracies,
            'nodes_pruned': nodes_pruned,
            'pruning_times': pruning_times
        }
    return results


# Plotting Function 
def plot_results(results):
    sizes = sorted(results.keys())
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # Colors and markers for methods
    colors = {'basic': 'blue', 'refined': 'green', 'v_structure': 'red', 'separators': 'orange', 'clusters': 'purple', 'full': 'cyan'}
    markers = {'basic': 'o', 'refined': 's', 'v_structure': 'd', 'separators': '^', 'clusters': '*', 'full': 'x'}

    # Plot costs, accuracies, nodes pruned, and pruning times
    measures = ['costs', 'accuracies', 'nodes_pruned', 'pruning_times']
    titles = ['Cost', 'Accuracy', 'Nodes Pruned (%)', 'Pruning Times (s)']

    for i, measure in enumerate(measures):
        ax = axs[i]
        for method in results[sizes[0]][measure]:
            mean_values = [np.mean(results[size][measure][method]) for size in sizes]
            std_values = [np.std(results[size][measure][method]) for size in sizes]

            # For Nodes Pruned, convert to percentage
            if measure == 'nodes_pruned':
                total_nodes = [size for size in sizes]
                mean_values = [100 * mean / total for mean, total in zip(mean_values, total_nodes)]

            ax.errorbar(sizes, mean_values, yerr=std_values,
                        fmt=markers[method], linestyle=':', color=colors[method],
                        ecolor=colors[method], elinewidth=1, capsize=3, capthick=1, alpha=0.7, label=f'{method.capitalize()} {measure}')

        ax.set_title(titles[i])
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel(titles[i])
        ax.legend()

    plt.tight_layout()
    plt.show()




# Parameters for the experiment
node_sizes = [10, 20, 30, 40, 50, 60, 70,80,90,100]  # Sizes of the networks to be tested
num_runs = 1  # Number of different runs for averaging
num_repeats = 10  # Number of repeats per run for more statistical accuracy
evidence_no = 0.8  # Proportion of nodes used as evidence
edge_den = 0.3  # Edge density
max_parents = 5  # Maximum number of parents any node can have
seed = 42  # Seed for reproducibility

# Running the experiment
experiment_results = run_experiment(node_sizes, num_runs, num_repeats, evidence_no, edge_den, max_parents, seed)

# Plotting the results
plot_results(experiment_results)


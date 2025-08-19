import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
from logger_setup import logger

def generate_map(city_name, output_file, image_file):
    """
    Generates a map from OpenStreetMap data for a given city and saves it as a GraphML file.
    Also, it plots the map and saves it as a PNG image.
    """
    logger.info(f"Generating map for {city_name}...")
    # Download the map data
    G = ox.graph_from_place(city_name, network_type='drive', simplify=True)

    # Save the graph to a file
    ox.save_graphml(G, filepath=output_file)
    logger.info(f"Graph for {city_name} saved to {output_file}")


    # Convert the MultiDiGraph to an undirected Graph for simpler plotting (optional)
    # This avoids plotting both directions of each edge

    plt.figure(figsize=(12, 8))
    G_simple = nx.Graph(G)  # This merges multi-edges and ignores direction
    pos = {node: (data['x'], data['y']) for node, data in G_simple.nodes(data=True)}
    nx.draw_networkx_edges(G_simple, pos, edge_color='gray', width=1)
    nx.draw_networkx_nodes(G_simple, pos, node_size=10, node_color='blue', alpha=0.7)
    plt.savefig(image_file, dpi=300, format='png')
    logger.info(f"Map image for {city_name} saved to {image_file}")


def main():
    # Define the city and output file path
    city_name = "Blida, Algeria"
    output_file = "data/Blida_map.graphml"
    image_file = "data/Blida_map.png"
    
    # Generate the map
    generate_map(city_name, output_file, image_file)

if __name__ == "__main__":
    main()

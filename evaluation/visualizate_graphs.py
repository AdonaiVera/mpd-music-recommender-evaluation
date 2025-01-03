import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx

def create_bar_plot(train_metrics, test_metrics, save_path="results/metrics_bar_plot.png"):
    """
    Create a bar plot for train and test metrics (R-Precision, NDCG, RSC).
    """
    labels = ["R-Precision", "NDCG", "RSC"]
    train_values = [train_metrics["r_precision"], train_metrics["ndcg"], train_metrics["rsc"]]
    test_values = [test_metrics["r_precision"], test_metrics["ndcg"], test_metrics["rsc"]]

    x = range(len(labels))  # Number of metrics

    # Create bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(x, train_values, width=0.4, label="Train", align="center")
    plt.bar([i + 0.4 for i in x], test_values, width=0.4, label="Test", align="center")

    # Add labels and legend
    plt.xticks([i + 0.2 for i in x], labels)
    plt.ylabel("Metric Value")
    plt.title("Training vs Testing Metrics")
    plt.ylim(0, 1)  # Assuming metrics range between 0 and 1
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    print(f"Bar plot saved at: {save_path}")
    plt.close()


def create_loss_auc_plot(self, train_losses, train_auc, val_auc, test_auc, num_playlists, save_path):
    """
    Create and save a polished and interactive visualization for loss and AUC metrics over epochs.
    """
    epochs = list(range(1, self.config["num_epochs"] + 1))

    fig = go.Figure()

    # Add traces for each metric
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_losses,
            mode="lines",
            name="Train Loss",
            line=dict(color="red", dash="dot"),
            hovertemplate="Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_auc,
            mode="lines+markers",
            name="Train AUC",
            line=dict(color="blue", width=2),
            marker=dict(size=6, symbol="circle"),
            hovertemplate="Epoch %{x}<br>Train AUC: %{y:.4f}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=val_auc,
            mode="lines+markers",
            name="Validation AUC",
            line=dict(color="green", width=2),
            marker=dict(size=6, symbol="square"),
            hovertemplate="Epoch %{x}<br>Validation AUC: %{y:.4f}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=test_auc,
            mode="lines+markers",
            name="Test AUC",
            line=dict(color="orange", width=2),
            marker=dict(size=6, symbol="triangle-up"),
            hovertemplate="Epoch %{x}<br>Test AUC: %{y:.4f}<extra></extra>"
        )
    )

    # Update layout for polished aesthetics
    fig.update_layout(
        title=dict(
            text=f"<b>Training Metrics Over Epochs</b><br><span style='font-size:14px'>Playlists: {num_playlists}</span>",
            x=0.5,
            y=0.9,
            font=dict(family="Arial", size=24)
        ),
        xaxis=dict(
            title="Epochs",
            titlefont=dict(size=16),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor="lightgrey"
        ),
        yaxis=dict(
            title="Metrics (Loss / AUC)",
            titlefont=dict(size=16),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor="lightgrey",
            range=[0, 1]  # Ensures values are always in a valid range
        ),
        width=1000,
        height=600,
        legend=dict(
            title="Legend",
            font=dict(size=12),
            orientation="h",
            y=-0.2,
            x=0.5,
            xanchor="center"
        ),
        margin=dict(l=40, r=40, t=80, b=60),
        hovermode="x unified",
        template="plotly_white"
    )

    # Add annotations to highlight significant points (optional)
    fig.add_annotation(
        x=epochs[-1],
        y=train_losses[-1],
        text=f"Final Loss: {train_losses[-1]:.4f}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40
    )
    fig.add_annotation(
        x=epochs[-1],
        y=test_auc[-1],
        text=f"Final Test AUC: {test_auc[-1]:.4f}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=40
    )

    # Config for interactivity
    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "staticPlot": False,
        "displaylogo": False
    }

    # Save the plot
    fig.write_html(save_path, config=config)
    print(f"Plot saved at: {save_path}")

    return fig, config

def plot_graph(edges, node_labels=None, save_path=None, title="Graph Visualization", 
               sample_size=None):
    """
    Plot a graph with nodes and edges using NetworkX, with optional edge sampling.

    :param edges: DataFrame or iterable of edges (source, target).
    :param node_labels: Optional dictionary of node labels (e.g., {node_id: "Label"}).
    :param save_path: Path to save the plot.
    :param title: Title for the graph.
    :param sample_size: Number of edges to randomly sample (default: include all).
    """
    # Sample edges if sample_size is provided
    if sample_size and sample_size < len(edges):
        edges = edges.sample(n=sample_size, random_state=42)

    # Create a graph object
    G = nx.Graph()

    # Add edges to the graph
    for _, row in edges.iterrows():
        G.add_edge(row["playlist_id"], row["song_id"])

    # Assign node colors based on type (playlist or song)
    node_colors = []
    for node in G.nodes():
        if node in edges["playlist_id"].unique():
            node_colors.append("lightgreen")  # Playlists: Green
        elif node in edges["song_id"].unique():
            node_colors.append("lightblue")  # Songs: Blue
        else:
            node_colors.append("gray")  # Default: Gray

    # Layout for the graph
    pos = nx.spring_layout(G, seed=42)  # Seed for consistent layout

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=(node_labels is not None),
        labels=node_labels,
        node_size=500,
        node_color=node_colors,
        edge_color="gray",
        font_size=10,
        font_color="black",
    )
    plt.title(title, fontsize=16)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Graph saved at: {save_path}")
    plt.show()

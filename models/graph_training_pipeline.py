import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
from methods.data_utils import DataManager

class GraphSAGELinkPrediction(nn.Module):
    def __init__(self, num_playlists, num_songs, playlist_dim, song_dim, dropout_prob):
        super(GraphSAGELinkPrediction, self).__init__()
        self.playlist_embedding = nn.Embedding(num_playlists, playlist_dim)
        self.song_embedding = nn.Embedding(num_songs, song_dim)
        self.fc = nn.Linear(playlist_dim + song_dim, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, playlist_ids, song_ids):
        playlist_embedded = self.playlist_embedding(playlist_ids)
        song_embedded = self.song_embedding(song_ids)
        concatenated_embeddings = torch.cat((playlist_embedded, song_embedded), dim=1)
        concatenated_embeddings = self.dropout(concatenated_embeddings)
        prediction = torch.sigmoid(self.fc(concatenated_embeddings))
        return prediction

class GraphTrainingPipeline:
    def __init__(self, cur_dir, model_arch, device, config):
        """
        Initialize the training pipeline with user-defined hyperparameters.
        :param cur_dir: Directory of the dataset
        :param model_arch: Model architecture to use (e.g., GraphSAGE)
        :param device: Device to run the model (CPU or GPU)
        :param config: Dictionary containing hyperparameters
        """
        self.cur_dir = cur_dir
        self.save_path="results/training_metrics.html"
        self.model_arch = model_arch
        self.device = device
        self.config = config
        self.data = DataManager(cur_dir)

    def train(self, return_model=False):
        """
        Main function to train the model and save results.
        """
        num_playlists =self.data.num_playlists()
        num_songs = self.data.num_songs()
        parsed_data = self.data.train_model_data()

        # Initialize the model
        if self.model_arch == "GraphSAGE":
            model = GraphSAGELinkPrediction(
                num_playlists,
                num_songs,
                self.config["playlist_embedding_dim"],
                self.config["song_embedding_dim"],
                self.config["dropout"],
            )

        trained_model, output_data = self._train_model(parsed_data, model)
        fig, config = self._create_loss_auc_plot(
            output_data["train_losses"],
            output_data["train_auc_scores"],
            output_data["val_auc_scores"],
            output_data["test_auc_scores"],
            num_playlists,
            save_path=self.save_path,
        )

        # Save results
        self.data.save_model_weights(trained_model)
        self.data.save_training_data(output_data)

        return trained_model if return_model else None

    def _train_model(self, parsed_data, model):
        """
        Core training loop with metrics tracking.
        """
        model = model.to(self.device)
        sample_weights = torch.where(
            parsed_data["train"]["label_ids"].to(self.device) == 1,
            torch.tensor(self.config["pos_edge_weight"], device=self.device),
            torch.tensor(self.config["neg_edge_weight"], device=self.device),
        )
        criterion = nn.BCEWithLogitsLoss(weight=sample_weights)
        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])

        metrics = {
            "train_losses": [],
            "train_auc_scores": [],
            "val_auc_scores": [],
            "test_auc_scores": [],
            "precision_scores": [],
            "recall_scores": [],
            "f1_scores": [],
        }

        for epoch in range(self.config["num_epochs"]):
            model.train()
            optimizer.zero_grad()

            outputs = model(
                parsed_data["train"]["playlist_ids"].to(self.device),
                parsed_data["train"]["song_ids"].to(self.device),
            )
            loss = criterion(
                outputs.view(-1),
                parsed_data["train"]["label_ids"].to(self.device, dtype=torch.float),
            )
            metrics["train_losses"].append(loss.item())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_auc = self._compute_auc(outputs, parsed_data["train"])
                val_auc = self._evaluate_auc(model, parsed_data["val"])
                test_metrics = self._evaluate_test_metrics(model, parsed_data["test"])

            metrics["train_auc_scores"].append(train_auc)
            metrics["val_auc_scores"].append(val_auc)
            metrics["test_auc_scores"].append(test_metrics["auc"])
            metrics["precision_scores"].append(test_metrics["precision"])
            metrics["recall_scores"].append(test_metrics["recall"])
            metrics["f1_scores"].append(test_metrics["f1"])

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.config['num_epochs']}], "
                    f"Loss: {loss.item():.4f}, Train AUC: {train_auc:.4f}, "
                    f"Validation AUC: {val_auc:.4f}, Test AUC: {test_metrics['auc']:.4f}, "
                    f"Precision: {test_metrics['precision']:.4f}, "
                    f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}"
                )

        return model, metrics


    def _compute_auc(self, outputs, data):
        """
        Compute AUC for predictions with single-class handling.
        """
        probs = outputs.cpu().view(-1).numpy()
        labels = data["label_ids"].numpy()

        if len(set(labels)) < 2:
            return None  
        return roc_auc_score(labels, probs)

    def _evaluate_auc(self, model, data):
        """
        Evaluate AUC on validation/test data.
        """
        model.eval()
        with torch.no_grad():
            outputs = model(
                data["playlist_ids"].to(self.device),
                data["song_ids"].to(self.device),
            )
        return self._compute_auc(outputs, data)
    
    def _evaluate_test_metrics(self, model, data):
        """
        Evaluate test metrics (AUC, Precision, Recall, F1).
        """
        model.eval()
        with torch.no_grad():
            outputs = model(
                data["playlist_ids"].to(self.device),
                data["song_ids"].to(self.device),
            )
            probs = outputs.cpu().view(-1).numpy()
            preds = (probs > 0.5).astype(int)

            return {
                "auc": roc_auc_score(data["label_ids"].numpy(), probs),
                "precision": precision_score(data["label_ids"].numpy(), preds),
                "recall": recall_score(data["label_ids"].numpy(), preds),
                "f1": f1_score(data["label_ids"].numpy(), preds),
            }

    def _create_loss_auc_plot(self, train_losses, train_auc, val_auc, test_auc, num_playlists, save_path):
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

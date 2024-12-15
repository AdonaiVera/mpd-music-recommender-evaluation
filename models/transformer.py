from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt
import nvtabular as nvt
from nvtabular import ops as nvt_ops
from nvtabular.loader.torch import TorchAsyncItr
from merlin.schema import Tags
import ast
import pandas as pd

# Define constants
MAX_SEQUENCE_LENGTH = 100
D_MODEL = 64

# Load and preprocess the dataset with NVTabular
data = pd.read_csv("data/processed/playlists_df.csv")
data["tracks"] = data["tracks"].apply(ast.literal_eval)
data['tracks_length'] = data['tracks'].apply(len)

# Define NVTabular workflow
workflow = nvt.Workflow(
    {
        "tracks": nvt_ops.Categorify(),
        "num_tracks": nvt_ops.Normalize(),
        "num_albums": nvt_ops.Normalize(),
        "num_followers": nvt_ops.Normalize(),
        "tracks_length": nvt_ops.Normalize(),
    }
)

# Apply workflow to dataset
dataset = nvt.Dataset(data)
workflow.fit(dataset)
processed_data = workflow.transform(dataset)

# Export the schema
schema = processed_data.schema

# Define the input module to process tabular input features
input_module = tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    continuous_projection=D_MODEL,
    aggregation="concat",
    masking="causal",
)

# Define the transformer configuration
transformer_config = tr.XLNetConfig.build(
    d_model=D_MODEL, n_head=4, n_layer=2, total_seq_length=MAX_SEQUENCE_LENGTH
)

# Build the model's body
body = tr.SequentialBlock(
    input_module,
    tr.MLPBlock([D_MODEL]),
    tr.TransformerBlock(transformer_config, masking=input_module.masking)
)

# Define evaluation metrics
metrics = [
    NDCGAt(top_ks=[20, 40], labels_onehot=True),
    RecallAt(top_ks=[20, 40], labels_onehot=True),
]

# Define the model's head with the prediction task
head = tr.Head(
    body,
    tr.NextItemPredictionTask(weight_tying=True, metrics=metrics),
    inputs=input_module,
)

# Compile the end-to-end model
model = tr.Model(head)

# Prepare the data loader
train_dataset = processed_data.to_iter_torch(batch_size=64, shuffle=True)
train_loader = TorchAsyncItr(train_dataset, batch_size=64)

# Train the model
trainer = tr.Trainer(model=model, max_epochs=10)
trainer.fit(train_loader)

# Evaluate the model
test_dataset = processed_data.to_iter_torch(batch_size=64, shuffle=False)
test_loader = TorchAsyncItr(test_dataset, batch_size=64)
metrics = trainer.evaluate(test_loader)

# Print metrics
print("Evaluation Metrics:", metrics)
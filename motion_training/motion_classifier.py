import torch.nn as nn
import torch
import numpy as np


class SemanticEncoder(nn.Module):
    def __init__(
        self, input_feats, num_frames,
        latent_dim, transformer_feedforward_dim,
        num_layers, num_heads, dropout,
        semantic_pool_type, out_dim
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.transformer_feedforward_dim = transformer_feedforward_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_frames = num_frames

        self.input_feats = input_feats
        self.semantic_pool_type = semantic_pool_type

        self.input_process = InputProcess(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(
            self.latent_dim, self.dropout
        )

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.transformer_feedforward_dim,
            dropout=self.dropout,
            activation="gelu"
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer,
            num_layers=self.num_layers
        )

        if self.semantic_pool_type == 'linear_time_layer':
            self.linear_time = nn.Linear(
                in_features=self.num_frames,
                out_features=1
            )

        self.regressor = torch.nn.Linear(latent_dim, out_dim)
        self.classifier_activation = nn.Softmax(dim=-1)

    def forward(self, x):

        x = self.input_process(x)

        x_seq = self.sequence_pos_encoder(x)  # [seqlen, bs, d]

        encoder_output = self.seqTransEncoder(x_seq)   # [seqlen, bs, d]

        output = encoder_output.transpose(2, 0)   # # [semdim, bs, seqlen]

        if self.semantic_pool_type == 'global_avg_pool':
            output = torch.mean(output, dim=-1).transpose(1, 0)
        elif self.semantic_pool_type == 'global_max_pool':
            output = torch.amax(output, dim=-1).transpose(1, 0)
        elif self.semantic_pool_type == 'linear_time_layer':
            output = self.linear_time(output).squeeze().transpose(1, 0)
        elif self.semantic_pool_type == 'gated_multi_head_attention_pooling':
            # This could be interesting
            raise Exception("Not implemented.")
        else:
            raise Exception("Pool type not implemented.")

        output = self.regressor(output)

        # TODO: add final activation function, e.g. softmax
        output = self.classifier_activation(output)

        return output


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nframes, njoints, nfeats = x.shape
        x = x.permute((1, 0, 2, 3)).reshape(nframes, bs, njoints * nfeats)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.prod(
            torch.arange(0, d_model, 2).float() *
            (-np.log(torch.tensor(10000.0)) / d_model)
        ))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


if __name__ == "__main__":
    num_joints = 22
    num_feats = 3
    # Maximal number of frames. For shorter recordings you may want
    # to repeat the last frame until this number of frames is reached.
    # Longer ones must be cut.
    num_frames = 100

    dummy_model = SemanticEncoder(
        input_feats=num_joints * num_feats,
        num_frames=num_frames,
        latent_dim=256,
        transformer_feedforward_dim=512,
        num_layers=8,
        num_heads=4,
        dropout=0.2,
        semantic_pool_type='global_avg_pool',
        out_dim=2
    )

    batch_size = 32
    dummy_motions = torch.randn(batch_size, num_joints, num_feats, num_frames)

    print(dummy_model)

    output = dummy_model(dummy_motions)

    print(output)
    print(output.shape)

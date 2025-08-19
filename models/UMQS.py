import torch
import torch.nn as nn
import torch.nn.functional as F

class UMQS(nn.Module):
    def __init__(self, num_queries=300, keep_queries=100, query_dim=256):
        super().__init__()
        self.keep_queries = keep_queries
        self.scorer = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.ReLU(),
            nn.Linear(query_dim, 1)
        )

    def forward(self, queries, refpoints):
        """
        queries: [B, N, C]
        refpoints: [B, N, 4]
        """
        scores = self.scorer(queries).squeeze(-1)  # [B, N]
        topk = torch.topk(scores, self.keep_queries, dim=1).indices  # [B, keep_queries]

        # Gather top-k queries and refpoints
        B = queries.shape[0]
        batch_indices = torch.arange(B, device=queries.device).unsqueeze(1).expand(-1, self.keep_queries)

        filtered_queries = queries[batch_indices, topk]     # [B, keep_queries, C]
        filtered_refpoints = refpoints[batch_indices, topk] # [B, keep_queries, 4]

        return filtered_queries, filtered_refpoints

#By HCX. 2023.3.27
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class AttentionMemoryNetwork(nn.Module):

    def __init__(self, num_vectors, embedding_dim, hop_count, self_a):
        super(AttentionMemoryNetwork, self).__init__()
        self.num_vectors = num_vectors
        self.embedding_dim = embedding_dim
        self.hop_count = hop_count
        self.self_a = self_a
        self.embedding = nn.Embedding(num_vectors, embedding_dim)
        self.hop_layers = []
        for i in range(hop_count):
            hop_layer = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU())
            self.hop_layers.append(hop_layer)
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, input_vectors, parameter_vector):
        input_embeddings = self.embedding(input_vectors)
        memory_vector = parameter_vector
        self_a = self.self_a
        for i in range(self.hop_count):
            attention_weights = torch.cosine_similarity(input_embeddings, memory_vector.unsqueeze(-1)).squeeze(-1)
            attention_weights = nn.functional.softmax(attention_weights, dim=1)
            query_vec = parameter_vector.unsqueeze(0)
            vecs = self_a
            d = query_vec.size(-1)
            scores = torch.matmul(vecs, query_vec.transpose(0, 1)) / torch.sqrt(torch.tensor(d))
            attention_weights1 = torch.nn.functional.softmax(scores, dim=-1)
            output_vec = torch.matmul(attention_weights1, query_vec)
            output_vec = self.embedding(output_vec.long())
            weighted_sum = torch.matmul(output_vec.transpose(1, 2), attention_weights.unsqueeze(-1)).squeeze(-1)
            hop_output = self.hop_layers[i](weighted_sum)
            memory_vector = memory_vector + hop_output
        output = self.output_layer(memory_vector)
        return output.squeeze(-1)

class VectorDataset(Dataset):

    def __init__(self, vectors):
        super(VectorDataset, self).__init__()
        self.vectors = vectors

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, index):
        return self.vectors[index]

input_vectors = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 1], [0, 1, 2, 2], [0, 1, 2, 4], [0, 2, 1, 1], [0, 2, 1, 2], [0, 2, 1, 3],
                              [0, 2, 1, 4],[0, 2, 2, 1], [0, 2, 2, 2], [0, 2, 2, 3], [0, 2, 2, 1], [0, 1, 2, 3],
                              [1, 2, 3, 3], [1, 2, 3, 2], [1, 1, 3, 3], [1, 2, 2, 2]])
parameter_vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
dataset = VectorDataset(input_vectors)
self_a=input_vectors.float()
dataloader = DataLoader(dataset, batch_size=17)
num_vectors, embedding_dim = input_vectors.size()
hop_count = 3
model = AttentionMemoryNetwork(num_vectors, embedding_dim, hop_count, self_a)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 50
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch, parameter_vector)
        loss = criterion(output, torch.ones(output.size()))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss={loss.item()}")
test_vectors = torch.tensor([[0, 1, 2, 0], [2, 1, 3, 0], [0, 2, 3, 1], [1, 1, 2, 0], [1, 2, 3, 4]])
with torch.no_grad():
    for batch in test_vectors.split(1):
        output = model(batch, parameter_vector)
        print(output[0])

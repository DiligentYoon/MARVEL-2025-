import torch
import torch.nn as nn
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U = U.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(U, dim=-1)  

        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k) 
        V = torch.matmul(v_flat, self.w_value).view(shape_v)  

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3)) 

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U) 

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(U, dim=-1)  
        heads = torch.matmul(attention, V)  
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            self.w_out.view(-1, self.embedding_dim)
        ).view(-1, n_query, self.embedding_dim)

        return out, attention  


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multiHeadAttention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w


class Encoder(nn.Module):
    def __init__(self, in_channel, in_features, out_features, map_input_shape):
        super(Encoder, self).__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            )   
        )

        self.cnn_output_size = _get_conv_output_size(self.encoder_cnn, in_channel, map_input_shape)
        self.mlp_output_size = out_features
        self.total_latent_size = self.cnn_output_size + self.mlp_output_size

        self.encoder_mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )

    def forward(self, obs):
        x1 = obs["map"]
        x2 = obs["numerical"]
        z_cnn = self.encoder_cnn(x1)
        z_mlp = self.encoder_mlp(x2.squeeze(1)).unsqueeze(0)

        return torch.cat((z_cnn, z_mlp), dim=1)



class PolicyNet(nn.Module):
    def __init__(self, in_channel_team, in_features_team, out_features_team,
                 in_channel_indi, in_features_indi, out_features_indi,
                 map_input_shape_team, map_input_shape_indi,
                 embedding_dim, action_dim):
        """
        PolicyNet 초기화
        :param in_channel_team: 팀 관측용 맵의 채널 수
        :param in_channel_indi: 개인 관측용 맵의 채널 수
        :param map_input_shape_team: 맵의 (높이, 너비) ex) (250, 250)
        :param in_features_team: 팀 관측용 수치 데이터의 피쳐 수
        :param in_features_indi: 개인 관측용 수치 데이터의 피쳐 수
        :param embedding_dim: 임베딩 레이어의 차원
        :param action_dim: 최종 행동의 차원
        """
        super(PolicyNet, self).__init__()
        
        # --- 각 Branch 정의 ---
        self.team_encoder = Encoder(in_channel_team, in_features_team, out_features_team, map_input_shape_team)
        self.indi_encoder = Encoder(in_channel_indi, in_features_indi, out_features_indi, map_input_shape_indi)

        # --- Total Branch (Actor Head) 정의 ---
        # 모든 잠재 벡터를 합친 크기를 입력으로 받음
        total_latent_size = self.team_encoder.total_latent_size + self.indi_encoder.total_latent_size
                            
        self.actor_head = nn.Sequential(
            nn.Linear(total_latent_size, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, action_dim),
            nn.Tanh() 
        )
    

    def forward(self, observation: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        신경망의 순전파 로직
        :param observation: 4개의 키를 가진 딕셔너리
               - "team_map": (batch, C_team, H, W) 크기의 텐서
               - "team_numerical": (batch, N_team_features) 크기의 텐서
               - "indi_map": (batch, C_indi, H, W) 크기의 텐서
               - "indi_numerical": (batch, N_indi_features) 크기의 텐서
        :return: (batch, action_dim) 크기의 행동 텐서
        """
        # Team Branch 처리
        z_team = self.team_encoder(observation["team"])
        
        # Individual Branch 처리
        z_indi = self.indi_encoder(observation["individual"])
        
        # Total Branch 처리
        z_total = torch.cat([z_team, z_indi], dim=1).squeeze(0)
        action = self.actor_head(z_total)
        
        return action




class QNet(nn.Module):
    def __init__(self, node_dim, embedding_dim, num_angles_bin, train_algo):
        super(QNet, self).__init__()

        # Graph encoder
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=4, n_layer=6)

        # Local frontiers distribution encoder
        self.frontiers_embedding = nn.Conv1d(num_angles_bin, embedding_dim, kernel_size=3, padding=1)
        self.node_frontiers_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # Decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # Heeading layer
        self.best_headings_embedding = nn.Linear(num_angles_bin, embedding_dim)
        self.visited_headings_embedding = nn.Linear(num_angles_bin, embedding_dim)
        self.neighboring_node_embedding = nn.Linear(embedding_dim * 3, embedding_dim)

        # Agent decoder
        if train_algo in (2 ,3):
            self.initial_embedding = nn.Linear(node_dim + 1, embedding_dim)
        else:
            # Graph embedding
            self.initial_embedding = nn.Linear(node_dim, embedding_dim)

        if train_algo in (1, 3):
            self.agent_decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
            self.all_agent_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

            self.q_values_layer = nn.Linear(embedding_dim * 3, 1)
        else:
            self.q_values_layer = nn.Linear(embedding_dim * 2, 1)


    def encode_graph(self, node_inputs, node_padding_mask, edge_mask, frontier_distribution):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature,
                                                         key_padding_mask=node_padding_mask,
                                                         attn_mask=edge_mask)
        
        frontier_distribution = frontier_distribution.permute(0, 2, 1)
        frontiers_feature = self.frontiers_embedding(frontier_distribution)
        frontiers_feature = frontiers_feature.permute(0, 2, 1)

        enhanced_node_feature = self.node_frontiers_embedding(torch.cat((enhanced_node_feature, frontiers_feature), dim=-1))

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                                  current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(current_node_feature,
                                                                    enhanced_node_feature,
                                                                    node_padding_mask)

        return current_node_feature, enhanced_current_node_feature

    def output_q(self, current_node_feature, enhanced_current_node_feature, enhanced_node_feature,
                 current_edge, edge_padding_mask, headings_visited, neighbor_best_headings, current_index, all_agent_indices, all_agent_next_indices):
        embedding_dim = enhanced_node_feature.size()[2]
        num_best_headings = neighbor_best_headings.size()[2]
        batch_size = enhanced_node_feature.size()[0]
        current_state_feature = self.current_embedding(torch.cat((enhanced_current_node_feature,
                                                                  current_node_feature), dim=-1))

        neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                           current_edge.repeat(1, 1, embedding_dim))
        
        enhanced_neighbor_best_headings = self.best_headings_embedding(neighbor_best_headings)
        all_headings_visited = self.visited_headings_embedding(headings_visited)
        all_neighbor_headings_visited = torch.gather(all_headings_visited, 1,
                                           current_edge.repeat(1, 1, embedding_dim))   

        neighboring_nodes_feature = neighboring_feature.unsqueeze(2).repeat(1, 1, num_best_headings, 1)
        neighbor_headings_visited = all_neighbor_headings_visited.unsqueeze(2).repeat(1, 1, num_best_headings, 1)

        enhanced_neighbor_features = self.neighboring_node_embedding(torch.cat((neighboring_nodes_feature, neighbor_headings_visited,
                                                                                enhanced_neighbor_best_headings), dim=-1)).reshape(batch_size, -1, embedding_dim)
        
        if all_agent_indices != None:
            all_agent_node_feature = torch.gather(enhanced_node_feature, 1,
                                                all_agent_indices.repeat(1, 1, embedding_dim))
            all_agent_selected_neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                                                all_agent_next_indices.repeat(1, 1, embedding_dim))

            all_agent_action_features = torch.cat((all_agent_node_feature, all_agent_selected_neighboring_feature), dim=-1)
            all_agent_action_features = self.all_agent_embedding(all_agent_action_features)

            agent_mask = all_agent_indices == current_index
            global_state_action_feature, _ = self.agent_decoder(current_state_feature, all_agent_action_features, agent_mask)
            action_features = torch.cat((current_state_feature.repeat(1, enhanced_neighbor_features.size()[1], 1), enhanced_neighbor_features, global_state_action_feature.repeat(1, enhanced_neighbor_features.size()[1], 1)), dim=-1)
            q_values = self.q_values_layer(action_features)
        else:
            action_features = torch.cat((current_state_feature.repeat(1, enhanced_neighbor_features.size()[1], 1), enhanced_neighbor_features), dim=-1)
            q_values = self.q_values_layer(action_features)
        return q_values

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask, frontier_distribution, headings_visited, neighbor_best_headings, all_agent_indices=None, all_agent_next_indices=None):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask, frontier_distribution)
        current_node_feature, enhanced_current_node_feature = self.decode_state(enhanced_node_feature, current_index, node_padding_mask)
        q_values = self.output_q(current_node_feature, enhanced_current_node_feature,
                                 enhanced_node_feature, current_edge, edge_padding_mask, headings_visited, neighbor_best_headings, current_index, all_agent_indices, all_agent_next_indices)
        return q_values


def _get_conv_output_size(encoder, in_channel, map_shape):
    """ CNN 인코더의 최종 출력 크기를 계산하기 위한 헬퍼 함수 """
    dummy_input = torch.randn(1, in_channel, *map_shape)
    output = encoder(dummy_input)
    return output.shape[1]
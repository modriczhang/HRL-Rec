"""
Model Hyper Parameter Dict

February 2020
modric10zhang@gmail.com

"""

param_dict = {
    'num_epochs': 10,  # training epoch
    'feat_dim': 32,  # feature embedding dimension
    'field_num': 9,  # number of user feature fields
    'user_field_num': 3,  # field number of user
    'doc_field_num': 3,  # field number of doc
    'con_field_num': 3,  # field number of context
    'max_clk_seq': 10,  # max click sequence
    'rnn_max_len': 10,  # max length of sequence in RNN
    'rnn_layer': 1,  # layer number of RNN
    'batch_size': 16,  # batch size
    'gamma': 0.3,  # discounted factor
    'double_networks_sync_step': 0.1,  # double networks sync step
    'double_networks_sync_freq': 30,  # double networks sync frequency
    'actor_explore_range': 0.01,  # exploration range for policy networks
    'encoder_layer': 1,  # encoder layer number
    'head_num': 4,  # head number for self-attention
    'lr': 0.0002,  # learning rate of network
    'dropout': 0.3,  # dropout ratio
    'grad_clip': 5.0,  # grad clip
}

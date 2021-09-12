
## HRL-Rec

Integrated recommendation aims to jointly recommend heterogeneous items in the main feed from different sources via multiple channels, which needs to capture user preferences on both item and channel levels. 
It has been widely used in practical systems by billions of users, while few works concentrate on the integrated recommendation systematically. 

In this work, we propose a novel Hierarchical reinforcement learning framework for integrated recommendation (HRL-Rec), which divides the integrated recommendation into two tasks to recommend channels and items sequentially. 

The low-level agent is a channel selector, which generates a personalized channel list. The high-level agent is an item recommender, which recommends specific items from heterogeneous channels under the channel constraints. 

HRL-Rec has also been deployed on WeChat Top Stories, affecting millions of users.

## Note

In the actual online system, HRL-Rec is a complex re-ranking framework implemented in C++. All models are trained based on a deeply customized version of distributed tensorflow supporting large-scale sparse features.

Without massive data and machine resources, training HRL-Rec is not realistic.

Therefore, the open source code here only implements a simplified version of its core ideas for the reference of interested researchers. If there are any errors, please contact me. Thanks!
# importing the libraries

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----

# hyperparameters
batch_size = 64
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
vocab_size= 65
n_embd= 32
n_head = 4
n_layer = 4
dropout = 0.2
# -------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8')as f :
    text = f.read()

# All the characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from string to int
stoi = {ch:i for i, ch in enumerate(chars)} # this is the dictionary for string to int
itos= {i:ch for i , ch in enumerate(chars)} # for i to strings
encode = lambda s: [stoi[ch] for ch in s ] # encoder gives out the encoded string in form of list
decode = lambda l: ''.join([itos[i] for i in l]) # decoder gives the decoded sttring



# now lets bring in the tensors
data = torch.tensor(encode(text), dtype= torch.long)

# spliting the dataset to train and test
n = int(0.9*(len(data)))
train_data=data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix])
    y = torch.stack([data[i+1:i+block_size+1 ]for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split]= losses.mean()
    model.train()
    return out

class Block(nn.Module):
    """ Transformer Block: communication followed y computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MutliHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Rsidual connection
        x = x + self.sa(self.ln1(x))
        x= x+ self.ffwd(self.ln2(x))
        return x


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embd ,4* n_embd),
                                 nn.ReLU(),
                                 nn.Linear(4*n_embd, n_embd),
                                 nn.Dropout(dropout)
                        )
    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self , head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias= True )
        self.query = nn.Linear(n_embd, head_size, bias = True )
        self.value = nn.Linear(n_embd , head_size, bias = True )
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward( self, x ):
        B,T,C =x.shape # B, T, C
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinites")
        wei = q @ k.transpose(-2, -1 )* (C**(-0.5)) # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim = -1) # (B,T,T)
        wei = self.dropout(wei) # (B,T,T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out= wei@v # (B,T,T) @ (B, T, C) --> (B,T,C)
        return out

class MutliHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # each token directly
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) :

        out =  torch.cat([h(x) for h in self.heads], dim =-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out




#super simple Bigram Model
class BigramLM(nn.Module):

    def __init__ (self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx , targets=None):
        B,T= idx.shape
        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        x= tok_emb+ pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) #(B,T, vocab_size)
        if targets is  None:
            loss =None
        else :
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # print(logits.shape, targets.shape)
            loss = F.cross_entropy(logits , targets )
        return logits, loss

    def generate(self , idx , max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the preds
            idx_cond = idx[:, -block_size:] # (B, T)j
            logits , loss = self(idx_cond)
            logits = logits[:,-1,:]# becomes (B,C)
            probs = F.softmax(logits, dim = 1 ) # get the probabilites
            idx_next = torch.multinomial(probs, num_samples = 1) # get the top one (B,1)
            idx = torch.cat((idx , idx_next ), dim =1 ) # (B,T+1)
        return idx

model = BigramLM()
model.to(device)


# create a PyTorch optimiser
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

for iter in range(max_iters):

    #every once in a while evaluate the loss on train and val sets
    if iter % eval_interval== 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}      valid loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the logits and loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#generate from the model
context =torch.zeros((1,1), dtype= torch.long, device = device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

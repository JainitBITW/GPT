with open('input.txt', 'r', encoding='utf-8')as f :
    text = f.read()

print(f'length of the dataset is {len(text)}')

# All the characters 
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)

# mapping from string to int
stoi = {ch:i for i, ch in enumerate(chars)} # this is the dictionary for string to int 
itos= {i:ch for i , ch in enumerate(chars)} # for i to strings 
encode = lambda s: [stoi[ch] for ch in s ] # encoder gives out the encoded string in form of list 
decode = lambda l: ''.join([itos[i] for i in l]) # decoder gives the decoded sttring

print(encode("Hi Jainit"))
print(decode(encode("hi Jainit")))

# now lets bring in the tensors 
import torch 
data = torch.tensor(encode(text), dtype= torch.long)
print(data[:100])

n = int(0.9*(len(data)))
train_data=data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y= train_data[1:block_size+1]
for t in range (block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when the input is {context}, the target is {target}")

torch.manual_seed(1337)
batch_size = 4
block_size = 8 

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix])
    y = torch.stack([data[i+1:i+block_size+1 ]for i in ix])
    
    return x, y

xb , yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets')
print(yb.shape)
print(yb)
print('----')

import torch.nn as nn 
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLM(nn.Module):

    def __init__ (self,vocab_size):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx , targets=None):
        # idx and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets== None:
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
            logits , loss = self(idx)
            logits = logits[:,-1,:]# becomes (B,C)
            probs = F.softmax(logits, dim = 1 ) # get the probabilites 
            idx_next = torch.multinomial(probs, num_samples = 1) # get the top one (B,1)
            idx = torch.cat((idx , idx_next ), dim =1 ) # (B,T+1)
        return idx 
        
m = BigramLM(vocab_size)
logits, loss = m(xb , yb)
print(logits.shape)
print(loss)
print(decode(m.generate(idx=torch.zeros((1,1), dtype = torch.long),max_new_tokens=100)[0].tolist()))

# create a PyTorch optimiser 
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

batch_size = 32
for steps in range(10000):
    xb, yb = get_batch('train')
    logits , loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())


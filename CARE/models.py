"""Graph-aligned semantic tokenizer and Sentence-T5-conditioned SID generator."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class SIDVocabulary:
    PAD, EOS = 0, 1
    def __init__(self,codes,state=None):
        self.codes={int(k):list(map(int,v)) for k,v in codes.items()}
        if state is None:
            length=len(next(iter(self.codes.values())))
            sizes=[max(256,1+max(v[i] for v in self.codes.values())) for i in range(length)]
            starts=[]; cursor=2
            for size in sizes: starts.append(cursor); cursor+=size
            self.state={'starts':starts,'sizes':sizes,'size':cursor}
        else: self.state=state
        self.paths={a:[s+x for s,x in zip(self.state['starts'],c)] for a,c in self.codes.items()}


def build_t5(vocab_size, dropout_rate=.1):
    config=T5Config(vocab_size=vocab_size,d_model=128,d_ff=1024,d_kv=64,num_heads=6,
        num_layers=4,num_decoder_layers=4,dropout_rate=dropout_rate,feed_forward_proj='relu',
        pad_token_id=0,eos_token_id=1,decoder_start_token_id=0,tie_word_embeddings=True,use_cache=True)
    return T5ForConditionalGeneration(config)


class SemanticGenerator(nn.Module):
    def __init__(self,vocab_size,query_dim=768,memory_tokens=8,dropout_rate=.1):
        super().__init__()
        self.t5=build_t5(vocab_size, dropout_rate)
        self.memory_tokens=memory_tokens
        self.query_projection=nn.Sequential(nn.Linear(query_dim,128*memory_tokens),nn.GELU(),
                                            nn.LayerNorm(128*memory_tokens))
        self.memory_position=nn.Parameter(torch.randn(1,memory_tokens,128)*.02)

    def encode(self,query_vectors):
        h=self.query_projection(query_vectors).view(-1,self.memory_tokens,128)+self.memory_position
        mask=torch.ones(h.shape[:2],dtype=torch.long,device=h.device)
        return BaseModelOutput(last_hidden_state=h),mask

    def sequence_scores(self,query_vectors,labels):
        encoded,mask=self.encode(query_vectors)
        decoder=self.t5._shift_right(labels)
        logits=self.t5(encoder_outputs=encoded,attention_mask=mask,decoder_input_ids=decoder,
                       use_cache=False,return_dict=True).logits
        valid=labels.ne(-100)
        target=labels.clamp_min(0)
        token=F.log_softmax(logits.float(),-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)
        return (token*valid).sum(-1)/valid.sum(-1).clamp_min(1),logits

    def loss(self,query_vectors,labels,positive_mask,groups,weights,rank_weight):
        scores,logits=self.sequence_scores(query_vectors,labels)
        token_loss=F.cross_entropy(logits[positive_mask].reshape(-1,logits.shape[-1]),
                                   labels[positive_mask].reshape(-1),ignore_index=-100,reduction='none')
        token_loss=token_loss.view(positive_mask.sum(),-1).mean(-1)
        generation=(token_loss*weights[positive_mask]).sum()/weights[positive_mask].sum().clamp_min(1e-8)
        rank=[]
        for pos,neg in groups:
            rank.append(F.softplus(scores[neg]-scores[pos]).mean())
        ranking=torch.stack(rank).mean() if rank else scores.sum()*0
        return generation+rank_weight*ranking,generation,ranking,scores


@torch.no_grad()
def generate(model,vocab,queries,device,top_k=20,beam=30,batch_size=16):
    lookup={tuple(v):a for a,v in vocab.paths.items()}
    children={}
    for path in lookup:
        for depth,token in enumerate(path): children.setdefault(path[:depth],set()).add(token)
        children.setdefault(path,set()).add(1)
    def allowed(_batch,ids):
        prefix=tuple(ids.tolist()[1:])
        if prefix and prefix[-1]==1: return [1]
        return sorted(children.get(prefix,{1}))
    result={}; keys=list(queries); model.eval(); length=len(next(iter(vocab.paths.values())))
    for start in range(0,len(keys),batch_size):
        batch=keys[start:start+batch_size]
        q=torch.stack([queries[k] for k in batch]).to(device)
        encoded,mask=model.encode(q)
        out=model.t5.generate(encoder_outputs=encoded,attention_mask=mask,num_beams=beam,
             num_return_sequences=top_k,max_new_tokens=length+1,min_new_tokens=length,
             early_stopping=True,length_penalty=0.,prefix_allowed_tokens_fn=allowed)
        out=out.reshape(len(batch),top_k,-1).cpu().tolist()
        for key,seqs in zip(batch,out):
            ranking=[]
            for seq in seqs:
                path=[]
                for token in seq[1:]:
                    if token==1: break
                    if token!=0: path.append(token)
                aid=lookup.get(tuple(path))
                if aid is None or aid in ranking: raise AssertionError('Constrained generation produced invalid/duplicate API')
                ranking.append(aid)
            result[key]=ranking
    return result


class LightGCN(nn.Module):
    def __init__(self,n_users,n_items,dimension,adjacency,content_dim=768,layers=3):
        super().__init__(); self.n_users=n_users; self.n_items=n_items; self.layers=layers
        self.embedding=nn.Embedding(n_users+n_items,dimension)
        nn.init.normal_(self.embedding.weight,std=.1)
        self.mapper=nn.Linear(dimension,content_dim)
        self.register_buffer('adjacency',adjacency)

    def propagate(self):
        current=self.embedding.weight; values=[current]
        for _ in range(self.layers):
            current=torch.sparse.mm(self.adjacency,current); values.append(current)
        result=torch.stack(values).mean(0)
        return result[:self.n_users],result[self.n_users:]

    def loss(self,users,pos,neg,api_content,seen):
        u,i=self.propagate()
        positive=(u[users]*i[pos]).sum(-1); negative=(u[users]*i[neg]).sum(-1)
        bpr=F.softplus(negative-positive).mean()
        mapped=F.normalize(self.mapper(i[seen]),dim=-1)
        align=(1-(mapped*api_content[seen]).sum(-1)).mean()
        regularization=1e-5*(self.embedding(users).pow(2).mean()+
                             self.embedding(self.n_users+pos).pow(2).mean()+
                             self.embedding(self.n_users+neg).pow(2).mean())
        return bpr+.2*align+regularization,bpr,align

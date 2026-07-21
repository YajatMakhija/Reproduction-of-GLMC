from typing import Optional, Tuple
from enums import MatrixType, SamplerType
import torch
import torch.nn as nn
from einops import rearrange
from utils import absorb_ln_scale, apply_mean_subtraction_to_weights, expand, interpolate, RMSNorm, project, project_to_attn_circuits, replace_layernorm
import copy
from transformers import PreTrainedModel, GPT2Config
from weight_matching import weight_matching
import random

class Conv1DMerger(nn.Module):

    def __init__(self, conv1d_0, conv1d_1):
        super().__init__()
        self.register_buffer("conv1d_0_weight", conv1d_0.weight.data.clone().contiguous())
        if conv1d_0.bias is not None:
            self.register_buffer("conv1d_0_bias", conv1d_0.bias.data.clone().contiguous())
        else:
            self.conv1d_0_bias = None

        self.register_buffer("conv1d_1_weight", conv1d_1.weight.data.clone().contiguous())
        if conv1d_1.bias is not None:
            self.register_buffer("conv1d_1_bias", conv1d_1.bias.data.clone().contiguous())
        else:
            self.conv1d_1_bias = None

        self.P_in = None
        self.P_out = None
        self.nf = conv1d_0.nf
        self.coeff = None
    
    def set_P_in(self, P_in):
        self.P_in = P_in
    
    def set_P_out(self, P_out):
        self.P_out = P_out
    
    def set_coeff(self, coeff: int):
        self.coeff = coeff

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)

        # Handle weight interpolation
        weight = interpolate(self.conv1d_0_weight, self.P_in @ self.conv1d_1_weight @ self.P_out, self.coeff)

        # Handle bias interpolation
        if self.conv1d_0_bias is not None and self.conv1d_1_bias is not None:
            bias = interpolate(self.conv1d_0_bias, self.conv1d_1_bias @ self.P_out, self.coeff)
        else:
            bias = None

        # Apply the linear operation
        x = x.view(-1, x.size(-1))
        if bias is not None:
            x = torch.addmm(bias, x, weight)
        else:
            x = torch.matmul(x, weight)
        
        x = x.view(size_out)
        return x

class LinearMerger(nn.Module):
    def __init__(self, conv1d_0, conv1d_1):
        super().__init__()
        self.register_buffer("conv1d_0_weight", conv1d_0.weight.data.t().clone().contiguous())
        if conv1d_0.bias is not None:
            self.register_buffer("conv1d_0_bias", conv1d_0.bias.data.t().clone().contiguous())
        else:
            self.conv1d_0_bias = None

        self.register_buffer("conv1d_1_weight", conv1d_1.weight.data.t().clone().contiguous())
        if conv1d_1.bias is not None:
            self.register_buffer("conv1d_1_bias", conv1d_1.bias.data.t().clone().contiguous())
        else:
            self.conv1d_1_bias = None

        self.P_in = None
        self.P_out = None
        self.nf = self.conv1d_0_weight.shape[1]
        self.nx = self.conv1d_0_weight.shape[0]
        self.coeff = None
    
    def set_coeff(self, coeff: int):
        self.coeff = coeff
    
    def set_P_in(self, P_in):
        self.P_in = P_in

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)

        # Handle weight interpolation
        weight = interpolate(self.conv1d_0_weight, self.P_in @ self.conv1d_1_weight, self.coeff)

        # Handle bias interpolation
        if self.conv1d_0_bias is not None and self.conv1d_1_bias is not None:
            bias = interpolate(self.conv1d_0_bias, self.conv1d_1_bias @ self.P_out, self.coeff)
        else:
            bias = None

        # Apply the linear operation
        x = x.view(-1, x.size(-1))
        if bias is not None:
            x = torch.addmm(bias, x, weight)
        else:
            x = torch.matmul(x, weight)
        
        x = x.view(size_out)
        return x


class Conv1DMergerCATTN(nn.Module):

    def __init__(self, conv1d_0, conv1d_1, num_heads, embed_dim):
        super().__init__()
        self.register_buffer("conv1d_0_weight", conv1d_0.weight.data.clone().contiguous())
        self.register_buffer("conv1d_0_bias", conv1d_0.bias.data.clone().contiguous())

        self.register_buffer("conv1d_1_weight", conv1d_1.weight.data.clone().contiguous())
        self.register_buffer("conv1d_1_bias", conv1d_1.bias.data.clone().contiguous())

        self.P_in = None
        self.P_out = None
        self.nf = conv1d_0.nf
        self.nx = conv1d_0.nx
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coeff = None
    
    def set_coeff(self, coeff: int):
        self.coeff = coeff
    
    def set_P_in(self, P_in):
        self.P_in = P_in
    
    def set_P_out(self, P_out):
        self.P_out = P_out
    
    def _permute_heads(self, weight, bias, P):
        def permute(A, P):
            return torch.matmul(P, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])
        c_attn_weight = torch.cat((weight.t(), bias.data.t().unsqueeze(1)), dim=1)

        Q,K,V = c_attn_weight.data.chunk(3, dim = 0)
        Q = torch.cat((Q[:,:-1] @ self.P_in.t(), Q[:,-1].unsqueeze(1)), dim=-1)

        Q = rearrange(Q, '(h d) m -> h d m', h = self.num_heads, m = self.embed_dim+1)
        K = rearrange(K, '(h d) m -> h d m', h = self.num_heads, m = self.embed_dim+1)
        V = rearrange(V, '(h d) m -> h d m', h = self.num_heads, m = self.embed_dim+1)

        Q = torch.cat((torch.bmm(Q.transpose(1, 2)[:,:,:-1], self.P_in.t().expand(self.num_heads, -1, -1)), Q.transpose(1, 2)[:,:,-1:]), dim=-1).transpose(1, 2)
        Q = permute(Q, P)
        K = permute(K, P)
        V = permute(V, P)

        Q = Q.reshape(-1, self.embed_dim+1)
        K = K.reshape(-1, self.embed_dim+1)
        V = V.reshape(-1, self.embed_dim+1)

        return torch.cat((Q,K,V), dim=0)[:,:-1].t(), torch.cat((Q,K,V), dim=0)[:,-1:].t()

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        conv1d_1_weight, conv1d_1_bias = self._permute_heads(self.conv1d_1_weight, self.conv1d_1_bias, self.P_out)

        bias = interpolate(self.conv1d_0_bias, conv1d_1_bias, self.coeff)
        weight = interpolate(self.conv1d_0_weight, conv1d_1_weight, self.coeff)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        x = x.view(size_out)
        return x

class Conv1DMergerCPROJ(nn.Module):

    def __init__(self, conv1d_0, conv1d_1, num_heads, embed_dim):
        super().__init__()
        self.register_buffer("conv1d_0_weight", conv1d_0.weight.data.clone().contiguous())
        self.register_buffer("conv1d_0_bias", conv1d_0.bias.data.clone().contiguous())

        self.register_buffer("conv1d_1_weight", conv1d_1.weight.data.clone().contiguous())
        self.register_buffer("conv1d_1_bias", conv1d_1.bias.data.clone().contiguous())

        self.P_in = None
        self.P_out = None
        self.nf = conv1d_0.nf
        self.nx = conv1d_0.nx
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coeff = None
    
    def set_coeff(self, coeff: int):
        self.coeff = coeff
    
    def set_P_in(self, P_in):
        self.P_in = P_in
    
    def set_P_out(self, P_out):
        self.P_out = P_out
    
    def _permute_heads(self, weight, bias, P):
        def permute(A, P):
            return torch.matmul(P, A.reshape(A.shape[0], -1)).reshape(A.shape[0], A.shape[1], A.shape[2])
        OUT = rearrange(weight.t(), ' m (h d) -> m h d', h = self.num_heads, m = self.embed_dim)
        OUT = OUT.permute(1, 2, 0)

        OUT = torch.cat((OUT.transpose(1,2)[:,:,:-1] @ self.P_out.expand(self.num_heads, -1, -1), OUT.transpose(1,2)[:,:,-1:]),dim=-1).transpose(1,2)

        OUT = permute(OUT, P)

        OUT = OUT.permute(2, 0, 1)
        OUT = OUT.reshape(self.embed_dim, -1)
        OUT = OUT.t()
        return OUT, bias

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        conv1d_1_weight, conv1d_1_bias = self._permute_heads(self.conv1d_1_weight @ self.P_out, self.conv1d_1_bias @ self.P_out, self.P_in)
        bias = interpolate(self.conv1d_0_bias, conv1d_1_bias, self.coeff)
        weight = interpolate(self.conv1d_0_weight, conv1d_1_weight, self.coeff)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        x = x.view(size_out)
        return x

class RMSMerger(nn.Module):
    def __init__(self, rmsnorm_0, rmsnorm_1):
        super().__init__()

        self.register_buffer("bias_0", rmsnorm_0.bias.data.clone().contiguous())
        self.register_buffer("bias_1", rmsnorm_1.bias.data.clone().contiguous())

        self.norm = RMSNorm(dim=rmsnorm_0.weight.shape[0], eps=rmsnorm_0.eps, bias=False)
        self.norm.weight = nn.Parameter(torch.ones(rmsnorm_0.weight.shape[0]))
        self.P = None

        for param in self.norm.parameters():
            param.requires_grad = False
        self.bias_0.required_grad = False
        self.bias_1.required_grad = False
        self.coeff = None
    
    def set_coeff(self, coeff: int):
        self.coeff = coeff
    
    def set_P(self, P):
        self.P = P

    def forward(self, x):
        x = self.norm(x)
        if self.bias_0 is not None:
            return x + interpolate(self.bias_0, self.P @ self.bias_1, coeff=self.coeff)


class EmbeddingMerger(nn.Module):
    def __init__(self, embedding_0, embedding_1):
        super().__init__()
        self.embedding_0 = copy.deepcopy(embedding_0)
        self.embedding_1 = copy.deepcopy(embedding_1)

        for param in self.embedding_0.parameters():
            param.requires_grad = False
        for param in self.embedding_1.parameters():
            param.requires_grad = False

        self.P = None
        self.coeff = None
    
    def set_coeff(self, coeff: int):
        self.coeff = coeff
    
    def set_P(self, P):
        self.P = P
    
    def forward(self, x):
        return interpolate(self.embedding_0(x), self.embedding_1(x) @ self.P, coeff=self.coeff)
    
    
class GPTMerger(nn.Module):
    def _absorb(self, model):
        batch_size = 2
        seq_length = 32  # can be up to 512
        dummy_input = torch.randint(0, model.lm_head.weight.shape[0], (batch_size, seq_length))
        outputs_init = model(input_ids=dummy_input).logits

        absorb_ln_scale(model)
        replace_layernorm(model)
        apply_mean_subtraction_to_weights(model)

        outputs_final = model(input_ids=dummy_input).logits

        assert torch.allclose(outputs_init, outputs_final, atol=1e-5), "outputs are not close!"

    def __init__(self, model0, model1, token_freqs=None, permutations_only=False, iterations=15):
        super().__init__()
        model0 = model0.eval()
        model1 = model1.eval()

        self._absorb(model0)
        self._absorb(model1)

        self._permutations_only = permutations_only


        def random_parameter(dim0, dim1=None):
            dim1 = dim0 if dim1 is None else dim1
            eye = torch.eye(dim0, dim1)
            noise_std = 1e-2
            noise = torch.randn_like(eye) * noise_std
            return nn.Parameter(eye + noise)
        
        embed_dim = model0.transformer.wte.weight.shape[1]
        num_heads = model0.transformer.h[0].attn.num_heads

        for i in range(len(model0.transformer.h)):
            project_to_attn_circuits(model0, i)
        
        batch_size = 2
        seq_length = 32  # can be up to 512
        dummy_input = torch.randint(0, model1.lm_head.weight.shape[0], (batch_size, seq_length))
        outputs_init = model1(input_ids=dummy_input).logits
        for i in range(len(model1.transformer.h)):
            project_to_attn_circuits(model1, i)
        outputs_final = model1(input_ids=dummy_input).logits
        assert torch.allclose(outputs_init, outputs_final, atol=1e-5), "outputs are not close!"

        assert model0.transformer.wte.weight.shape[1] >= model1.transformer.wte.weight.shape[1], "Model 0 cannot be larger than model 1. Swap them for width-heterogeneous merging."
        if model0.transformer.wte.weight.shape[1] > model1.transformer.wte.weight.shape[1]:
            assert not permutations_only, "Permutation alignment is not supported for width heterogeneous merging."
            batch_size = 2
            seq_length = 32  # can be up to 512
            dummy_input = torch.randint(0, model1.lm_head.weight.shape[0], (batch_size, seq_length))
            outputs_init = model1(input_ids=dummy_input).logits
            model1 = expand(model1, model0.transformer.wte.weight.shape[1])
            outputs_final = model1(input_ids=dummy_input).logits
            assert torch.allclose(outputs_init, outputs_final, atol=1e-5), "outputs are not close!"

            outputs_init = model1(input_ids=dummy_input).logits
            for i in range(len(model1.transformer.h)):
                project_to_attn_circuits(model1, i)
            outputs_final = model1(input_ids=dummy_input).logits
            assert torch.allclose(outputs_init, outputs_final, atol=1e-5), "outputs are not close!"

        weight_matching(model0, model1, heads=num_heads, iterations=iterations, token_freqs=token_freqs, permutations_only=permutations_only)

        self.proj = nn.ParameterDict({
            "residual": random_parameter(model0.transformer.wte.weight.shape[1], model1.transformer.wte.weight.shape[1])
        })

        for i in range(len(model0.transformer.h)):
            self.proj[f"attention_heads_{i}"] = random_parameter(num_heads)
            self.proj[f"mlp_{i}"] = random_parameter(model0.transformer.h[i].mlp.c_fc.bias.shape[0])

        self.model = copy.deepcopy(model0)

        self.model.transformer.wte = EmbeddingMerger(model0.transformer.wte, model1.transformer.wte)
        self.model.transformer.wpe = EmbeddingMerger(model0.transformer.wpe, model1.transformer.wpe)

        for i in range(len(self.model.transformer.h)):
            self.model.transformer.h[i].ln_1 = RMSMerger(model0.transformer.h[i].ln_1, model1.transformer.h[i].ln_1)

            self.model.transformer.h[i].attn.c_attn = Conv1DMergerCATTN(model0.transformer.h[i].attn.c_attn, model1.transformer.h[i].attn.c_attn, num_heads=num_heads, embed_dim=embed_dim)
            self.model.transformer.h[i].attn.c_proj = Conv1DMergerCPROJ(model0.transformer.h[i].attn.c_proj, model1.transformer.h[i].attn.c_proj, num_heads=num_heads, embed_dim=embed_dim)

            self.model.transformer.h[i].mlp.c_fc = Conv1DMerger(model0.transformer.h[i].mlp.c_fc, model1.transformer.h[i].mlp.c_fc)
            self.model.transformer.h[i].mlp.c_proj = Conv1DMerger(model0.transformer.h[i].mlp.c_proj, model1.transformer.h[i].mlp.c_proj)

            self.model.transformer.h[i].ln_2 = RMSMerger(model0.transformer.h[i].ln_2, model1.transformer.h[i].ln_2)

        self.model.transformer.ln_f =  RMSMerger(model0.transformer.ln_f, model1.transformer.ln_f)
        self.model.lm_head = LinearMerger(model0.lm_head, model1.lm_head)

        self.set_sampler(sampler_type=None)
    
    def set_sampler(self, sampler_type: str, fixed_coeff=0.5):
        if sampler_type is None:
            self._sampler = lambda: fixed_coeff
        else:
            sampler_type = sampler_type.lower()
            if sampler_type == SamplerType.GAUSSIAN.value:
                self._sampler = lambda: min(max(random.gauss(0.5, 0.1), 0.0), 1.0)
            elif sampler_type == SamplerType.UNI.value:
                self._sampler = lambda: random.uniform(0.0, 1.0)
            elif sampler_type == SamplerType.NARROW_UNI.value:
                self._sampler = lambda: random.uniform(0.4, 0.6)
            elif sampler_type == SamplerType.NARROW_UNI_BIASED.value:
                self._sampler = lambda: random.uniform(0.2, 0.5)
            else:
                raise ValueError(f"Unknown sampler type: {sampler_type!r}")

    
    def _project(self, coeff: float):
        P_res = project(self.proj["residual"], matrix_type=MatrixType.PERM if self._permutations_only else MatrixType.ORTHO)

        self.model.transformer.wte.set_P(P_res)
        self.model.transformer.wte.set_coeff(coeff)
        self.model.transformer.wpe.set_P(P_res)
        self.model.transformer.wpe.set_coeff(coeff)

        for i in range(len(self.model.transformer.h)):
            self.model.transformer.h[i].ln_1.set_P(P_res.t())
            self.model.transformer.h[i].ln_1.set_coeff(coeff)

            self.model.transformer.h[i].attn.c_attn.set_P_in(P_res.t())
            self.model.transformer.h[i].attn.c_attn.set_coeff(coeff)

            P_heads = project(self.proj[f"attention_heads_{i}"], matrix_type=MatrixType.PERM)
            self.model.transformer.h[i].attn.c_attn.set_P_out(P_heads)
            self.model.transformer.h[i].attn.c_attn.set_coeff(coeff)

            self.model.transformer.h[i].attn.c_proj.set_P_out(P_res)
            self.model.transformer.h[i].attn.c_proj.set_P_in(P_heads)
            self.model.transformer.h[i].attn.c_proj.set_coeff(coeff)

            self.model.transformer.h[i].mlp.c_fc.set_P_in(P_res.t())
            self.model.transformer.h[i].mlp.c_fc.set_coeff(coeff)

            P_mlp = project(self.proj[f"mlp_{i}"], matrix_type=MatrixType.PERM)
            self.model.transformer.h[i].mlp.c_fc.set_P_out(P_mlp)
            self.model.transformer.h[i].mlp.c_fc.set_coeff(coeff)

            self.model.transformer.h[i].mlp.c_proj.set_P_out(P_res)
            self.model.transformer.h[i].mlp.c_proj.set_P_in(P_mlp.t())
            self.model.transformer.h[i].mlp.c_proj.set_coeff(coeff)

            self.model.transformer.h[i].ln_2.set_P(P_res.t())
            self.model.transformer.h[i].ln_2.set_coeff(coeff)

        self.model.transformer.ln_f.set_P(P_res.t())
        self.model.transformer.ln_f.set_coeff(coeff)
        self.model.lm_head.set_P_in(P_res.t())
        self.model.lm_head.set_coeff(coeff)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        coeff = self._sampler()
        self._project(coeff=coeff)
        # 🔥 Ensure tensors are on the right device
        device = next(self.parameters()).device

        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)
        if labels is not None:
            labels = labels.to(device)

        return self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class GPTMergerWrapper(PreTrainedModel):
    def __init__(self, config: GPT2Config, merger_model: GPTMerger):
        super().__init__(config)
        self.merger_model = merger_model
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.merger_model.forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    def to(self, device):
        self.merger_model = self.merger_model.to(device)
        return super().to(device)

    def state_dict(self, *args, **kwargs):
        return self.merger_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.merger_model.load_state_dict(state_dict, strict=strict)


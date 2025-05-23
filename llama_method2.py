import time

import torch
import torch.nn as nn

# from gptq_action1 import *
from gptq_action2 import *
from modelutils import *
from quant_testChao import *
# from quant import *


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    ele_sum_container = {'value': 0.0}
    R_aver_container = {'value': 0.0}
    R_record_container = {'value': torch.zeros((len(layers), 7))}

    quantizers = {}

    R_ini = torch.tensor([
    [0.0547, 0.0718, 3.1521, 1.1108, 2.8157, 2.5460, 3.1340],
        [0.6292, 0.3250, 3.2727, 2.9077, 2.9944, 3.3617, 3.2993],
        [1.1780, 1.0200, 3.5208, 3.1917, 3.1350, 1.0034, 3.5081],
        [1.6611, 1.2576, 3.6648, 2.8716, 3.2439, 3.9906, 3.6077],
        [2.0286, 1.8162, 3.6782, 3.2661, 3.0825, 3.6532, 3.6311],
        [2.1980, 2.0212, 4.0002, 3.2957, 3.1406, 4.0429, 3.9946],
        [2.2397, 2.0190, 4.0081, 3.3447, 3.2097, 4.0585, 3.9985],
        [2.2107, 2.0244, 4.0042, 3.5908, 3.1169, 4.0264, 3.9990],
        [2.4480, 2.0818, 4.0081, 3.5708, 3.0613, 4.0146, 3.9995],
        [2.6672, 2.1279, 4.0056, 3.7275, 3.0840, 4.0146, 3.9827],
        [2.9910, 2.2153, 4.0093, 3.6670, 3.0962, 4.0411, 3.9883],
        [2.7671, 2.0352, 4.0105, 3.7700, 3.0474, 4.0273, 3.7766],
        [3.0093, 2.3147, 4.0107, 3.8074, 3.0293, 4.0420, 3.9675],
        [3.0090, 2.6365, 4.0095, 3.7246, 3.1248, 4.0752, 3.9993],
        [3.0095, 2.4912, 4.0100, 3.8079, 3.2349, 4.0922, 4.0015],
        [3.0095, 2.5054, 4.0105, 3.8218, 3.2520, 4.0850, 4.0002],
        [3.0076, 2.5266, 4.0090, 3.5945, 3.1516, 4.0873, 3.9949],
        [3.0076, 2.7437, 4.0071, 3.7092, 3.1985, 4.0829, 3.9993],
        [3.0073, 2.7744, 4.0068, 3.6841, 3.3086, 4.0867, 3.9995],
        [3.0068, 2.7607, 4.0063, 3.6587, 3.3254, 4.0783, 3.9993],
        [3.0056, 2.8245, 4.0054, 3.5505, 3.3062, 4.0799, 3.9990],
        [3.0059, 2.8811, 4.0051, 3.3936, 3.3782, 4.0811, 3.9985],
        [3.0061, 3.0032, 4.0032, 3.2720, 3.5452, 4.0833, 3.9985],
        [3.0066, 2.9988, 4.0049, 3.0415, 3.5830, 4.0815, 3.9980],
        [3.0066, 3.0034, 4.0042, 3.3752, 3.6482, 4.0728, 3.9978],
        [3.0063, 3.0029, 4.0037, 3.1990, 3.5652, 4.0883, 3.9978],
        [3.0081, 3.0032, 4.0032, 3.1138, 3.5698, 4.0809, 3.9978],
        [3.0071, 3.0024, 4.0012, 3.0928, 3.4707, 4.0492, 3.9624],
        [3.0054, 2.9983, 4.0034, 2.9739, 3.3774, 4.0045, 3.5610],
        [3.0049, 3.0012, 4.0027, 3.0574, 3.2412, 3.7605, 3.1616],
        [3.0042, 2.9446, 3.9980, 2.8921, 3.0476, 2.8815, 3.0076],
        [2.9526, 2.0977, 3.7922, 2.0803, 2.7278, 1.4281, 2.9141]])
    # alpha
    R_ref = 3.0
    alpha = 2 ** (2 * R_ini - 2 * R_ref)
    
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            name2idx = {name: idx for idx, name in enumerate(subset)}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups, layer_idx=i, name = name, R_aver_container=R_aver_container, ele_sum_container=ele_sum_container, R_record_container=R_record_container, col_idx=name2idx[name], alpha=alpha
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    # ===========================================
    print(f"Final R_aver = {R_aver_container['value']:.3f}")
    print(f"Total elements processed = {ele_sum_container['value']:.0f}")
    print("R_record matrix:")
    print(R_record_container['value'])
    # col_weights = torch.tensor([768*768]*4 + [768*2300]*2)
    # # 计算加权总和
    # weighted_sum = torch.sum(R_record_container['value'] * col_weights)
    # print(f"col_weights.shape = {col_weights.shape}")
    # print(f"R_record_container['value'].shape = {R_record_container['value'].shape}")
    # # 总权重值（即参数总数）
    # total_weight = R_record_container['value'].shape[0] * torch.sum(col_weights)
    # # 最终加权平均
    # weighted_avg = weighted_sum / total_weight
    # print(f"Weighted average of R_record: {weighted_avg.item():.4f}")
    # ===========================================
    
    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


if __name__ == '__main__':
    import argparse
    from datautils_offline import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    args = parser.parse_args()

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    datasets = ['wikitext2', 'ptb', 'c4'] 
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)

    if args.save:
        llama_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)

